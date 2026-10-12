"""P0 long-video -> transcript -> reviewable edit plan -> MP4 APIs.

Independent of the existing storyboard and generative V2V endpoints. Intended
for a single trusted local/API host; use auth and per-user quotas before SaaS.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .edit_executor import MAX_SOURCE_BYTES, probe_media, render_edit_plan
from .edit_plan import (
    TranscriptSegment, EditPlan, PlanPatch, PlanRequest,
    draft_plan, validate_transcript,
)


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    READY = "READY"
    FAILED = "FAILED"


class RenderJob(BaseModel):
    id: str
    plan_version: int
    status: TaskStatus = TaskStatus.QUEUED
    error: str | None = None
    metrics: dict | None = None


class P0Project(BaseModel):
    id: str
    duration_ms: int
    source_has_audio: bool
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    transcript_origin: str = "none"
    plans: list[EditPlan] = Field(default_factory=list)
    renders: list[RenderJob] = Field(default_factory=list)


class TranscriptUpdate(BaseModel):
    segments: list[TranscriptSegment] = Field(min_length=1, max_length=10000)


class _Conflict(RuntimeError):
    pass


class EditingMVP:
    def __init__(self, root: Path, *, render_edge: int = 1080):
        self.root = (root / "p0_projects").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.render_edge = render_edge
        self.locks: dict[str, asyncio.Lock] = {}
        self.tasks: set[asyncio.Task] = set()
        self.render_gate = asyncio.Semaphore(1)

    def path(self, project_id: str) -> Path:
        try:
            if str(UUID(project_id)) != project_id:
                raise ValueError("not canonical UUID")
        except (ValueError, TypeError, AttributeError) as exc:
            raise KeyError("Project not found") from exc
        return self.root / project_id

    def lock(self, project_id: str) -> asyncio.Lock:
        return self.locks.setdefault(project_id, asyncio.Lock())

    def get(self, project_id: str) -> P0Project:
        try:
            return P0Project.model_validate_json(
                (self.path(project_id) / "record.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise KeyError("Project not found") from exc

    def save(self, project: P0Project) -> None:
        folder = self.path(project.id)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / (".record-" + uuid4().hex + ".tmp")
        try:
            tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, folder / "record.json")
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def busy(project: P0Project) -> bool:
        return any(r.status in {TaskStatus.RUNNING, TaskStatus.QUEUED} for r in project.renders)

    def recover(self) -> None:
        for file in self.root.glob("*/record.json"):
            try:
                record = self.get(file.parent.name)
                changed = False
                for job in record.renders:
                    if job.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                        job.status = TaskStatus.FAILED
                        job.error = "Process restarted; this render can be resubmitted"
                        changed = True
                if changed:
                    self.save(record)
            except (KeyError, ValueError, OSError):
                continue

    async def upload(self, upload: UploadFile) -> P0Project:
        if not upload.filename or Path(upload.filename).suffix.lower() != ".mp4":
            raise ValueError("Only .mp4 uploads accepted")
        record_id = str(uuid4())
        folder = self.path(record_id)
        folder.mkdir()
        source = folder / "source.mp4"
        try:
            total = 0
            with source.open("wb") as out:
                while block := await upload.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_SOURCE_BYTES:
                        raise ValueError("Source exceeds 1 GiB limit")
                    out.write(block)
            with source.open("rb") as f:
                header = f.read(12)
            if len(header) < 12 or header[4:8] != b"ftyp":
                raise ValueError("Invalid MP4 signature")
            duration_ms, has_audio, _, _ = await asyncio.to_thread(probe_media, source)
            record = P0Project(id=record_id, duration_ms=duration_ms, source_has_audio=has_audio)
            self.save(record)
            return record
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await upload.close()

    async def set_transcript(self, project_id: str, update: TranscriptUpdate) -> P0Project:
        async with self.lock(project_id):
            record = self.get(project_id)
            if self.busy(record):
                raise _Conflict("Cannot alter transcript during render")
            validate_transcript(update.segments, record.duration_ms)
            record.transcript = update.segments
            record.transcript_origin = "manual"
            # A transcript change invalidates previous approvals, not historical plans.
            for plan in record.plans:
                plan.approval = "pending"
            self.save(record)
            return record

    async def transcribe(self, project_id: str) -> P0Project:
        if find_spec("faster_whisper") is None:
            raise RuntimeError("ASR not installed: pip install -e '.[asr]'")
        async with self.lock(project_id):
            record = self.get(project_id)
            if self.busy(record):
                raise _Conflict("Cannot transcribe during render")
            if not record.source_has_audio:
                raise ValueError("Source has no audio; provide manual transcript")
            def run_asr():
                from faster_whisper import WhisperModel
                model_name = os.getenv("AVG_ASR_MODEL", "small")
                device = os.getenv("AVG_ASR_DEVICE", "cpu")
                compute = "int8" if device == "cpu" else "float16"
                model = WhisperModel(model_name, device=device, compute_type=compute)
                segments, _ = model.transcribe(
                    str(self.path(project_id) / "source.mp4"),
                    vad_filter=True, beam_size=5,
                )
                return [
                    TranscriptSegment(
                        start_ms=max(0, round(s.start * 1000)),
                        end_ms=round(s.end * 1000),
                        text=s.text.strip(),
                    ) for s in segments if s.text.strip() and s.end > s.start
                ]
            segments = await asyncio.to_thread(run_asr)
            if not segments:
                raise ValueError("ASR produced no speech; check source or supply manual transcript")
            # Some ASR model segments can overlap by a few ms; normalize without inventing speech.
            normalized = []
            for row in segments:
                start = max(normalized[-1].end_ms if normalized else 0, row.start_ms)
                end = min(record.duration_ms, row.end_ms)
                if end > start:
                    normalized.append(TranscriptSegment(start_ms=start, end_ms=end, text=row.text))
            validate_transcript(normalized, record.duration_ms)
            record.transcript = normalized
            record.transcript_origin = "faster-whisper"
            for plan in record.plans:
                plan.approval = "pending"
            self.save(record)
            return record

    async def make_plan(self, project_id: str, request: PlanRequest) -> EditPlan:
        async with self.lock(project_id):
            record = self.get(project_id)
            if self.busy(record):
                raise _Conflict("Cannot plan during render")
            plan = draft_plan(
                record.transcript, record.duration_ms,
                request, version=len(record.plans) + 1,
            )
            record.plans.append(plan)
            self.save(record)
            return plan

    async def patch_plan(self, project_id: str, version: int, patch: PlanPatch) -> EditPlan:
        async with self.lock(project_id):
            record = self.get(project_id)
            if self.busy(record):
                raise _Conflict("Cannot modify plan during render")
            if not record.plans or record.plans[-1].version != version:
                raise _Conflict("Stale plan version; fetch latest plan before editing")
            current = record.plans[-1]
            plan = EditPlan(
                version=version + 1, instruction=current.instruction,
                aspect=patch.aspect, clips=patch.clips, captions=patch.captions,
                planner="manual", approval="pending",
            ).validate_for_source(record.duration_ms)
            record.plans.append(plan)
            self.save(record)
            return plan

    async def approve(self, project_id: str, version: int) -> EditPlan:
        async with self.lock(project_id):
            record = self.get(project_id)
            if self.busy(record):
                raise _Conflict("Cannot approve during render")
            if not record.plans or record.plans[-1].version != version:
                raise _Conflict("Cannot approve stale plan")
            plan = record.plans[-1].validate_for_source(record.duration_ms)
            plan.approval = "approved"
            self.save(record)
            return plan

    async def enqueue(self, project_id: str, version: int) -> RenderJob:
        async with self.lock(project_id):
            record = self.get(project_id)
            if not record.plans or record.plans[-1].version != version:
                raise _Conflict("Render requires the latest plan")
            plan = record.plans[-1].validate_for_source(record.duration_ms)
            if plan.approval != "approved":
                raise _Conflict("Approve the latest plan before rendering")
            for job in record.renders:
                if job.plan_version == version and job.status in {
                    TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.READY,
                }:
                    return job  # idempotent: never render a successful version twice
            if self.busy(record):
                raise _Conflict("Another render is running")
            job = RenderJob(id=str(uuid4()), plan_version=version)
            record.renders.append(job)
            self.save(record)
            task = asyncio.create_task(self._render(project_id, job.id, plan.model_copy(deep=True)))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return job

    async def _render(self, project_id: str, job_id: str, plan: EditPlan) -> None:
        folder = self.path(project_id)
        output = folder / (job_id + ".mp4")
        async with self.render_gate:
            record = self.get(project_id)
            job = next(x for x in record.renders if x.id == job_id)
            job.status = TaskStatus.RUNNING
            self.save(record)
            try:
                metrics = await asyncio.to_thread(
                    render_edit_plan, folder / "source.mp4", plan, output,
                    short_edge=self.render_edge,
                )
                record = self.get(project_id)
                job = next(x for x in record.renders if x.id == job_id)
                job.status = TaskStatus.READY
                job.metrics = metrics
                job.error = None
            except asyncio.CancelledError:
                record = self.get(project_id)
                job = next(x for x in record.renders if x.id == job_id)
                job.status = TaskStatus.FAILED
                job.error = "Render cancelled"
                raise
            except Exception as exc:
                record = self.get(project_id)
                job = next(x for x in record.renders if x.id == job_id)
                job.status = TaskStatus.FAILED
                job.error = str(exc)[:600]
            finally:
                if job.status != TaskStatus.READY:
                    output.unlink(missing_ok=True)
                self.save(record)

    async def close(self) -> None:
        for job in list(self.tasks):
            job.cancel()
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def get_render(self, project_id: str, render_id: str) -> RenderJob:
        rec = self.get(project_id)
        try:
            if str(UUID(render_id)) != render_id:
                raise ValueError
        except (ValueError, TypeError):
            raise KeyError("Render not found") from None
        return next((j for j in rec.renders if j.id == render_id), None) or self._not_found()

    @staticmethod
    def _not_found():
        raise KeyError("Render not found")


def create_p0_router(mvp: EditingMVP) -> APIRouter:
    router = APIRouter(prefix="/api/mvp", tags=["P0 Editable Video"])
    def safe(call, *args):
        try:
            return call(*args)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/capabilities")
    async def capabilities():
        return {"asr_installed": find_spec("faster_whisper") is not None,
                "transcript_manual_fallback": True, "planner": "rules",
                "requires_human_approval": True, "v2v": False}

    @router.post("/projects", response_model=P0Project, status_code=201)
    async def create_project(file: UploadFile = File(...)):
        try:
            return await mvp.upload(file)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/projects/{project_id}", response_model=P0Project)
    async def project(project_id: str):
        return safe(mvp.get, project_id)

    @router.get("/projects/{project_id}/source")
    async def source(project_id: str):
        safe(mvp.get, project_id)
        return FileResponse(mvp.path(project_id) / "source.mp4", media_type="video/mp4")

    @router.post("/projects/{project_id}/transcribe", response_model=P0Project)
    async def transcribe(project_id: str):
        try:
            return await mvp.transcribe(project_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.put("/projects/{project_id}/transcript", response_model=P0Project)
    async def update_transcript(project_id: str, update: TranscriptUpdate):
        try:
            return await mvp.set_transcript(project_id, update)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/projects/{project_id}/plans", response_model=EditPlan)
    async def make_plan(project_id: str, request: PlanRequest):
        try:
            return await mvp.make_plan(project_id, request)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/projects/{project_id}/plans/{version}", response_model=EditPlan)
    async def get_plan(project_id: str, version: int):
        record = safe(mvp.get, project_id)
        return next((p for p in record.plans if p.version == version), None) or safe(
            lambda: (_ for _ in ()).throw(KeyError("Plan not found"))
        )

    @router.patch("/projects/{project_id}/plans/{version}", response_model=EditPlan)
    async def patch(project_id: str, version: int, changes: PlanPatch):
        try:
            return await mvp.patch_plan(project_id, version, changes)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/projects/{project_id}/plans/{version}/approve", response_model=EditPlan)
    async def approve(project_id: str, version: int):
        try:
            return await mvp.approve(project_id, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/projects/{project_id}/renders", response_model=RenderJob, status_code=202)
    async def enqueue(project_id: str, version: int):
        try:
            return await mvp.enqueue(project_id, version)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except _Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/projects/{project_id}/renders/{render_id}", response_model=RenderJob)
    async def render_status(project_id: str, render_id: str):
        return safe(mvp.get_render, project_id, render_id)

    @router.get("/projects/{project_id}/renders/{render_id}/file")
    async def render_file(project_id: str, render_id: str, download: bool = False):
        job = safe(mvp.get_render, project_id, render_id)
        if job.status != TaskStatus.READY:
            raise HTTPException(409, "Render is not verified READY")
        path = mvp.path(project_id) / (job.id + ".mp4")
        if not path.is_file():
            raise HTTPException(404, "Rendered MP4 not found")
        return FileResponse(
            path, media_type="video/mp4", filename="edited-short.mp4",
            content_disposition_type="attachment" if download else "inline",
        )

    return router
