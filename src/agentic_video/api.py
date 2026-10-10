"""FastAPI project lifecycle endpoints; no job broker required for this MVP."""
import asyncio
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
from .vision import (
    OpenAICompatibleVisionAnalyzer, VisionConfig, VisionError,
    VideoUnderstandingResult,
)


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


def create_app(
    service: VideoService | None = None,
    vision_analyzer: OpenAICompatibleVisionAnalyzer | None = None,
) -> FastAPI:
    service = service or build_service()
    # Model configuration and API credentials are server-side only.
    cfg = Settings.from_env()
    analyzer = vision_analyzer
    if analyzer is None and cfg.vision_model:
        analyzer = OpenAICompatibleVisionAnalyzer(VisionConfig(
            base_url=cfg.vision_url,
            model=cfg.vision_model,
            api_key=cfg.vision_api_key,
            max_frames=cfg.vision_frames,
            timeout_seconds=cfg.vision_timeout_seconds,
        ))
    vision_limiter = asyncio.Semaphore(2)
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
    app.state.vision_analyzer = analyzer

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
            "video_understanding": analyzer is not None,
            "vision_model": analyzer.config.model if analyzer else None,
            "vision_model_support_verified": False,
            "vision_frames_sent": analyzer.config.max_frames if analyzer else 0,
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

    @app.post("/api/video-understand", response_model=VideoUnderstandingResult)
    async def understand_video(
        file: UploadFile = File(...),
        instruction: str = Form("描述视频中的主要场景和动作，并拆分为分镜"),
        shot_count: int = Form(3),
        theme_style: str = Form("cinematic, consistent color palette"),
    ):
        if analyzer is None:
            raise HTTPException(
                status_code=409,
                detail="尚未配置视觉模型；请设置 AVG_VISION_MODEL 和服务端 API 地址",
            )
        if not 1 <= shot_count <= service.max_shots:
            raise HTTPException(status_code=422, detail="分镜数量超出配置上限")
        try:
            async with vision_limiter:
                plan, times = await analyzer.analyze(
                    file, instruction, shot_count, theme_style
                )
            project = service.create_from_shots(
                summary=plan.summary,
                style=theme_style, specs=plan.shots,
            )
            return VideoUnderstandingResult(
                summary=plan.summary,
                observations=plan.observations,
                sampled_seconds=times,
                project=project.model_dump(mode="json"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except VisionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
