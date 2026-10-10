"""Single-process orchestration; project mutation serialized with per-project locks."""
import asyncio
import logging
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import (
    CreateProjectRequest, Project, ProjectStatus, Shot, ShotStatus, UpdateShotRequest,
)
from .store import FileProjectStore
from .parsers import StoryboardParser

log = logging.getLogger(__name__)


class VideoBackend(Protocol):
    async def generate(self, prompt: str, destination: Path, duration: float) -> None: ...


class VideoComposer(Protocol):
    def compose(self, clips: list[tuple[Path, float]], output: Path) -> None: ...


class WorkflowConflict(RuntimeError):
    pass


class VideoService:
    def __init__(self, store: FileProjectStore, parser: StoryboardParser,
                 backend: VideoBackend, composer: VideoComposer,
                 concurrency: int = 1, max_shots: int = 5,
                 job_timeout_seconds: int = 1800):
        self.store, self.parser = store, parser
        self.backend, self.composer = backend, composer
        self.max_shots = max_shots
        self.job_timeout_seconds = job_timeout_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()

    def _lock(self, project_id: str) -> asyncio.Lock:
        return self._locks.setdefault(project_id, asyncio.Lock())

    def get(self, project_id: str) -> Project:
        return self.store.load(project_id)

    async def create(self, request: CreateProjectRequest) -> Project:
        if request.shot_count > self.max_shots:
            raise ValueError(f"shot_count exceeds configured maximum {self.max_shots}")
        specs = await self.parser.parse(
            request.script, request.theme_style, request.shot_count
        )
        if not specs or len(specs) > self.max_shots:
            raise ValueError("Storyboard parser returned invalid shot count")
        project = Project(
            project_id=str(uuid4()), script=request.script,
            theme_style=request.theme_style,
            shots=[Shot(id=i, **spec.model_dump()) for i, spec in enumerate(specs, 1)]
        )
        self.store.save(project)
        return project

    async def edit(self, project_id: str, shot_id: int,
                   request: UpdateShotRequest) -> Project:
        async with self._lock(project_id):
            project = self.store.load(project_id)
            if project.status in {ProjectStatus.GENERATING, ProjectStatus.STITCHING}:
                raise WorkflowConflict("Cannot edit while a project job is running")
            shot = self._shot(project, shot_id)
            shot.prompt = request.prompt
            if request.duration is not None:
                shot.duration = request.duration
            if request.camera_motion is not None:
                shot.camera_motion = request.camera_motion
            shot.revision += 1
            shot.status = ShotStatus.PENDING
            shot.error = None
            project.status = ProjectStatus.DRAFT
            project.final_video_path = None  # Old cut must not be served as current.
            project.error = None
            self.store.save(project)
            return project

    async def enqueue(self, project_id: str, shot_id: int | None = None) -> Project:
        async with self._lock(project_id):
            project = self.store.load(project_id)
            if project.status in {ProjectStatus.GENERATING, ProjectStatus.STITCHING}:
                raise WorkflowConflict("A project job is already running")
            if shot_id is None:
                shots = [
                    s for s in project.shots
                    if s.status != ShotStatus.READY or s.rendered_revision != s.revision
                ]
            else:
                shot = self._shot(project, shot_id)
                shot.revision += 1  # Explicit retry renders a new artifact/version.
                shots = [shot]
                project.final_video_path = None
            ids = [s.id for s in shots]
            for shot in shots:
                shot.status, shot.error = ShotStatus.GENERATING, None
            project.status = ProjectStatus.GENERATING if ids else ProjectStatus.STITCHING
            project.error = None
            self.store.save(project)
            task = asyncio.create_task(self._run(project_id, ids))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return project

    @staticmethod
    def _shot(project: Project, shot_id: int) -> Shot:
        for shot in project.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError("shot not found")

    async def _run(self, project_id: str, shot_ids: list[int]) -> None:
        active_id = None
        try:
            directory = self.store.directory(project_id)
            for shot_id in shot_ids:
                active_id = shot_id
                project = self.store.load(project_id)
                shot = self._shot(project, shot_id)
                name = f"shot-{shot.id}-r{shot.revision}-{uuid4().hex[:8]}.mp4"
                destination = directory / name
                try:
                    async with self._semaphore:
                        await asyncio.wait_for(
                            self.backend.generate(
                                project.theme_style + ". " + shot.prompt,
                                destination, shot.duration
                            ),
                            timeout=self.job_timeout_seconds,
                        )
                    if not destination.is_file() or not destination.stat().st_size:
                        raise RuntimeError("Backend did not produce a video file")
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise
                shot.video_path = name
                shot.rendered_revision = shot.revision
                shot.status = ShotStatus.READY
                shot.error = None
                self.store.save(project)
            project = self.store.load(project_id)
            if not all(
                s.status == ShotStatus.READY and
                s.revision == s.rendered_revision and s.video_path
                for s in project.shots
            ):
                project.status = ProjectStatus.DRAFT
                self.store.save(project)
                return
            project.status = ProjectStatus.STITCHING
            self.store.save(project)
            clips = [(self.store.asset(project_id, s.video_path), s.duration)
                     for s in project.shots]
            final_name = f"final-{uuid4().hex}.mp4"
            await asyncio.to_thread(
                self.composer.compose, clips, directory / final_name
            )
            project.final_video_path = final_name
            project.status = ProjectStatus.READY
            project.error = None
            self.store.save(project)
            self._remove_orphans(project)
        except asyncio.CancelledError:
            self._fail(project_id, active_id, "Generation cancelled")
            raise
        except Exception as exc:
            log.exception("Project %s failed", project_id)
            self._fail(project_id, active_id, str(exc)[:700])

    def _fail(self, project_id: str, active_id: int | None, message: str) -> None:
        project = self.store.load(project_id)
        project.status = ProjectStatus.FAILED
        project.error = message
        if active_id is not None:
            shot = self._shot(project, active_id)
            shot.status = ShotStatus.FAILED
            shot.error = message
        for shot in project.shots:
            if shot.status == ShotStatus.GENERATING:
                shot.status = ShotStatus.PENDING
        self.store.save(project)

    def _remove_orphans(self, project: Project) -> None:
        keep = {s.video_path for s in project.shots if s.video_path}
        if project.final_video_path:
            keep.add(project.final_video_path)
        for asset in self.store.directory(project.project_id).glob("*.mp4"):
            if asset.name not in keep:
                asset.unlink(missing_ok=True)

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
