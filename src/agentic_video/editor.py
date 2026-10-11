"""Single-host video-to-video edit jobs, separate from storyboard generation.

Edit jobs require an Omni model that actually supports video conditioning.
Upload validation prevents arbitrary-path input and unbounded local storage.
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class EditStatus(str, Enum):
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    READY = "READY"
    FAILED = "FAILED"


class EditJob(BaseModel):
    job_id: str
    prompt: str = Field(min_length=1, max_length=3000)
    status: EditStatus = EditStatus.QUEUED
    source_path: str = "source.mp4"
    output_path: str | None = None
    duration: float = 0
    error: str | None = None


class EditingUnavailable(RuntimeError):
    pass


def probe_video(path: Path) -> float:
    """Check that the uploaded MP4 contains an actual video stream."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type", "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if proc.returncode:
        raise ValueError("Unable to decode uploaded MP4: " + proc.stderr[-160:])
    try:
        data = json.loads(proc.stdout)
        duration = float(data["format"]["duration"])
        has_video = any(
            item.get("codec_type") == "video"
            for item in data.get("streams", [])
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Uploaded video metadata is invalid") from exc
    if not has_video or not 0 < duration <= 30:
        raise ValueError("Please upload a playable MP4 between 0 and 30 seconds")
    return duration


class VideoEditService:
    MAX_BYTES = 100 * 1024 * 1024

    def __init__(self, root: Path, backend, enabled: bool,
                 generation_semaphore: asyncio.Semaphore,
                 timeout_seconds: int = 1800):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.enabled = enabled
        self.semaphore = generation_semaphore
        self.timeout_seconds = timeout_seconds
        self.tasks: set[asyncio.Task] = set()

    def directory(self, job_id: str) -> Path:
        try:
            if str(UUID(job_id)) != job_id:
                raise ValueError("noncanonical id")
        except (ValueError, AttributeError) as exc:
            raise KeyError("edit job not found") from exc
        return self.root / job_id

    def load(self, job_id: str) -> EditJob:
        try:
            return EditJob.model_validate_json(
                (self.directory(job_id) / "job.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise KeyError("edit job not found") from exc

    def save(self, job: EditJob):
        directory = self.directory(job.job_id)
        tmp = directory / f".job-{uuid4().hex}.tmp"
        try:
            tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, directory / "job.json")
        finally:
            tmp.unlink(missing_ok=True)

    def recover_interrupted(self):
        for path in self.root.glob("*/job.json"):
            try:
                job = self.load(path.parent.name)
                if job.status in {EditStatus.QUEUED, EditStatus.GENERATING}:
                    job.status = EditStatus.FAILED
                    job.error = "API restarted; please submit a new edit"
                    self.save(job)
            except (KeyError, ValueError, OSError):
                log.warning("Unable to recover video edit at %s", path, exc_info=True)

    def asset(self, job_id: str, variant: str) -> Path:
        job = self.load(job_id)
        if variant == "source":
            name = job.source_path
        elif variant == "result" and job.status == EditStatus.READY:
            name = job.output_path
        else:
            raise KeyError("video asset not ready")
        if name not in {"source.mp4", "result.mp4"}:
            raise KeyError("invalid video asset")
        path = self.directory(job_id) / name
        if not path.is_file():
            raise KeyError("video asset not found")
        return path

    async def create(self, file: UploadFile, prompt: str) -> EditJob:
        if not self.enabled:
            raise EditingUnavailable(
                "视频智能编辑需要启用 AVG_BACKEND=omni，"
                "并部署支持 V2V (Video-to-Video) input_reference 的模型；Mock 模式不提供 AI 编辑。"
            )
        prompt = prompt.strip()
        if not 1 <= len(prompt) <= 3000:
            raise ValueError("编辑指令长度应为 1～3000 字符")
        if not file.filename or Path(file.filename).suffix.lower() != ".mp4":
            raise ValueError("仅接受 .mp4 文件")
        job = EditJob(job_id=str(uuid4()), prompt=prompt)
        directory = self.directory(job.job_id)
        directory.mkdir()
        source = directory / "source.mp4"
        try:
            total = 0
            with source.open("wb") as dest:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.MAX_BYTES:
                        raise ValueError("视频文件不能超过 100 MiB")
                    dest.write(chunk)
            with source.open("rb") as incoming:
                first = incoming.read(16)
            if len(first) < 12 or first[4:8] != b"ftyp":
                raise ValueError("上传内容不是有效的 MP4 文件")
            job.duration = await asyncio.to_thread(probe_video, source)
            if hasattr(self.backend, "validate_duration"):
                self.backend.validate_duration(job.duration)
            self.save(job)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await file.close()
        task = asyncio.create_task(self._run(job.job_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return job

    async def _run(self, job_id: str):
        job = self.load(job_id)
        output = self.directory(job_id) / "result.mp4"
        try:
            async with self.semaphore:
                job.status = EditStatus.GENERATING
                self.save(job)
                await asyncio.wait_for(
                    self.backend.generate(
                        job.prompt, output, job.duration,
                        input_reference=self.directory(job_id) / "source.mp4",
                    ), timeout=self.timeout_seconds
                )
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError("The model did not return a video")
            job.output_path = "result.mp4"
            job.status = EditStatus.READY
            job.error = None
        except asyncio.CancelledError:
            job.status = EditStatus.FAILED
            job.error = "Edit cancelled"
            raise
        except Exception as exc:
            log.exception("Video edit %s failed", job_id)
            job.status = EditStatus.FAILED
            job.error = str(exc)[:600]
        finally:
            if job.status != EditStatus.READY:
                output.unlink(missing_ok=True)
            self.save(job)

    async def close(self):
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
