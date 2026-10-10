"""FastAPI project lifecycle endpoints; no job broker required for this MVP."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from .backends import MockVideoBackend, VllmOmniVideoBackend
from .composer import FFmpegComposer
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.store.recover_interrupted()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="Agentic Video Generation", version="0.1.0", lifespan=lifespan
    )
    app.state.video_service = service

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
    async def get_video(project_id: str):
        project = guard(service.get, project_id)
        if project.status != ProjectStatus.READY or not project.final_video_path:
            raise HTTPException(409, detail="Current cut is not ready")
        path = guard(service.store.asset, project_id, project.final_video_path)
        return FileResponse(path, media_type="video/mp4", filename="rough-cut.mp4")

    @app.get("/api/projects/{project_id}/shots/{shot_id}/video")
    async def get_shot_video(project_id: str, shot_id: int):
        project = guard(service.get, project_id)
        shot = guard(service._shot, project, shot_id)
        if (
            shot.status != ShotStatus.READY or
            shot.revision != shot.rendered_revision or not shot.video_path
        ):
            raise HTTPException(409, detail="Current shot is not ready")
        path = guard(service.store.asset, project_id, shot.video_path)
        return FileResponse(path, media_type="video/mp4", filename=f"shot-{shot_id}.mp4")

    return app


def main():
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
