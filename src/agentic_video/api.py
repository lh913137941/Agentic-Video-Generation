"""FastAPI project lifecycle endpoints; no job broker required for this MVP."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from .backends import MockVideoBackend, VllmOmniVideoBackend
from .composer import FFmpegComposer
from .editor import EditingUnavailable, EditJob, VideoEditService
from .config import Settings
from .models import (
    CreateProjectRequest, Project, ProjectStatus, ShotStatus, UpdateShotRequest,
)
from .parsers import OpenAIStoryboardParser, RuleStoryboardParser
from .service import VideoService, WorkflowConflict
from .store import FileProjectStore


def build_service(settings: Settings | None = None) -> VideoService:
    cfg = settings or Settings.from_env()
    store = FileProjectStore(Path(cfg.data_dir))
    parser = (RuleStoryboardParser() if cfg.parser == "rule" else
              OpenAIStoryboardParser(cfg.llm_url, cfg.llm_model, cfg.llm_api_key))
    backend = (
        MockVideoBackend(cfg.model_fps, cfg.width, cfg.height)
        if cfg.backend == "mock" else
        VllmOmniVideoBackend(
            cfg.omni_url, cfg.width, cfg.height,
            cfg.model_fps, cfg.model_steps, cfg.job_timeout_seconds,
            frame_stride=cfg.model_frame_stride
        )
    )
    composer = FFmpegComposer(
        cfg.output_width, cfg.output_height, cfg.output_fps
    )
    return VideoService(
        store, parser, backend, composer,
        cfg.generation_concurrency, cfg.max_shots, cfg.job_timeout_seconds
    )


def create_app(service: VideoService | None = None) -> FastAPI:
    service = service or build_service()
    editor = VideoEditService(
        service.store.root / "video_edits",
        service.backend,
        enabled=isinstance(service.backend, VllmOmniVideoBackend),
        generation_semaphore=service._semaphore,
        timeout_seconds=service.job_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.store.recover_interrupted()
        editor.recover_interrupted()
        try:
            yield
        finally:
            await editor.close()
            await service.close()

    app = FastAPI(
        title="Agentic Video Generation", version="0.1.0", lifespan=lifespan
    )
    app.state.video_service = service
    app.state.video_editor = editor

    def guard(fn, *args):
        try:
            return fn(*args)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkflowConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def async_guard(coro):
        try:
            return await coro
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkflowConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}


    @app.get("/api/capabilities")
    async def capabilities():
        return {
            "storyboard_generation": True,
            "video_edit_upload": editor.enabled,
            "video_edit_backend": (
                "vllm-omni-v2v" if editor.enabled else "unavailable-in-mock"
            ),
            "model_v2v_support_verified": False,
            "spatial_masks": False,
            "temporal_range_control": False,
            "video_edit_note": (
                "The selected vLLM-Omni model must support input_reference "
                "video conditioning; model-specific support and temporal "
                "preservation are not guaranteed."
            ),
        }

    @app.post("/api/video-edits", response_model=EditJob, status_code=202)
    async def submit_video_edit(
        file: UploadFile = File(...), prompt: str = Form(...)
    ):
        try:
            return await editor.create(file, prompt)
        except EditingUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/video-edits/{job_id}", response_model=EditJob)
    async def get_video_edit(job_id: str):
        return guard(editor.load, job_id)

    @app.get("/api/video-edits/{job_id}/source")
    async def get_video_edit_source(job_id: str, download: bool = False):
        return FileResponse(
            guard(editor.asset, job_id, "source"), media_type="video/mp4",
            filename="original.mp4",
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/api/video-edits/{job_id}/result")
    async def get_video_edit_result(job_id: str, download: bool = False):
        job = guard(editor.load, job_id)
        if job.status.value != "READY":
            raise HTTPException(status_code=409, detail="Edited video is not ready")
        return FileResponse(
            guard(editor.asset, job_id, "result"), media_type="video/mp4",
            filename="edited-video.mp4",
            content_disposition_type="attachment" if download else "inline",
        )

    @app.post("/api/projects", response_model=Project, status_code=201)
    async def create_project(request: CreateProjectRequest):
        return await async_guard(service.create(request))

    @app.get("/api/projects/{project_id}", response_model=Project)
    async def get_project(project_id: str):
        return guard(service.get, project_id)

    @app.patch("/api/projects/{project_id}/shots/{shot_id}", response_model=Project)
    async def update_shot(
        project_id: str, shot_id: int, request: UpdateShotRequest
    ):
        return await async_guard(service.edit(project_id, shot_id, request))

    @app.post("/api/projects/{project_id}/generate",
              response_model=Project, status_code=202)
    async def generate_all(project_id: str):
        return await async_guard(service.enqueue(project_id))

    @app.post("/api/projects/{project_id}/shots/{shot_id}/regenerate",
              response_model=Project, status_code=202)
    async def regenerate(project_id: str, shot_id: int):
        return await async_guard(service.enqueue(project_id, shot_id))

    @app.get("/api/projects/{project_id}/video")
    async def get_video(project_id: str, download: bool = False):
        project = guard(service.get, project_id)
        if project.status != ProjectStatus.READY or not project.final_video_path:
            raise HTTPException(409, detail="Current cut is not ready")
        path = guard(service.store.asset, project_id, project.final_video_path)
        return FileResponse(path, media_type="video/mp4", filename="rough-cut.mp4",
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/projects/{project_id}/shots/{shot_id}/video")
    async def get_shot_video(project_id: str, shot_id: int, download: bool = False):
        project = guard(service.get, project_id)
        shot = guard(service._shot, project, shot_id)
        if (
            shot.status != ShotStatus.READY or
            shot.revision != shot.rendered_revision or not shot.video_path
        ):
            raise HTTPException(409, detail="Current shot is not ready")
        path = guard(service.store.asset, project_id, shot.video_path)
        return FileResponse(path, media_type="video/mp4", filename=f"shot-{shot_id}.mp4",
                            content_disposition_type="attachment" if download else "inline")

    return app


def main():
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
