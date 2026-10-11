"""Real asynchronous vLLM-Omni adapter and deterministic FFmpeg-based Mock."""
import asyncio
import hashlib
import os
from pathlib import Path
import httpx


class BackendError(RuntimeError):
    pass


class MockVideoBackend:
    """Generate actual, silent H.264 clips without GPU or external inference."""
    def __init__(self, fps: int = 16, width: int = 832, height: int = 480):
        self.fps, self.width, self.height = fps, width, height

    async def generate(self, prompt: str, destination: Path, duration: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        color = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:6]
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"color=c=0x{color}:s={self.width}x{self.height}:r={self.fps}",
            "-t", str(duration), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart",
            str(destination),
        ]
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        if proc.returncode or not destination.is_file() or not destination.stat().st_size:
            raise BackendError("Mock FFmpeg failed: " + stderr.decode(errors="replace")[-700:])


class VllmOmniVideoBackend:
    """POST multipart /v1/videos, poll, then stream /content to a local MP4."""
    def __init__(self, base_url: str, width: int, height: int,
                 fps: int, steps: int, timeout_seconds: int = 1800,
                 poll_interval: float = 2.0, frame_stride: int = 4):
        self.base_url = base_url.rstrip("/")
        self.width, self.height, self.fps, self.steps = width, height, fps, steps
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.frame_stride = frame_stride

    async def generate(self, prompt: str, destination: Path, duration: float,
                       input_reference: Path | None = None) -> None:
        """Optional MP4 input_reference requires a V2V-capable deployed model."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Not every diffusion model supports arbitrary duration; num_frames is explicit.
        # Model-specific frame-count rules must be configured for the deployed model.
        target_frames = max(1, round(duration * self.fps))
        num_frames = (max(1, round((target_frames - 1) / self.frame_stride) * self.frame_stride + 1)
                      if self.frame_stride else target_frames)
        fields = {
            "prompt": (None, prompt),
            "width": (None, str(self.width)),
            "height": (None, str(self.height)),
            "fps": (None, str(self.fps)),
            "num_frames": (None, str(num_frames)),
            "num_inference_steps": (None, str(self.steps)),
        }
        job_id = None
        complete = False
        tmp = destination.with_name(destination.stem + ".partial.mp4")
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60, read=120), follow_redirects=True
        ) as client:
            try:
                if input_reference is not None:
                    with input_reference.open("rb") as uploaded:
                        payload = dict(fields)
                        payload["input_reference"] = (
                            "source.mp4", uploaded, "video/mp4"
                        )
                        response = await client.post(
                            self.base_url + "/v1/videos", files=payload
                        )
                else:
                    response = await client.post(
                        self.base_url + "/v1/videos", files=fields
                    )
                response.raise_for_status()
                job_id = response.json().get("id")
                if not isinstance(job_id, str) or not job_id:
                    raise BackendError("vLLM-Omni did not return a video job id")
                while True:
                    if asyncio.get_running_loop().time() > deadline:
                        raise BackendError(f"vLLM-Omni video job timed out ({job_id})")
                    status_response = await client.get(
                        self.base_url + f"/v1/videos/{job_id}"
                    )
                    status_response.raise_for_status()
                    metadata = status_response.json()
                    status = metadata.get("status")
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled", "canceled", "expired"}:
                        raise BackendError(f"vLLM-Omni job {job_id}: {status}")
                    if status not in {"queued", "in_progress"}:
                        raise BackendError(f"Unknown vLLM-Omni job state: {status!r}")
                    await asyncio.sleep(self.poll_interval)
                async with client.stream(
                    "GET", self.base_url + f"/v1/videos/{job_id}/content"
                ) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as output:
                        async for chunk in response.aiter_bytes():
                            output.write(chunk)
                if not tmp.is_file() or tmp.stat().st_size == 0:
                    raise BackendError("vLLM-Omni returned an empty video")
                os.replace(tmp, destination)
                complete = True
            finally:
                tmp.unlink(missing_ok=True)
                if job_id and not complete:
                    # Best effort: abort on timeout, failure, and coroutine cancellation.
                    try:
                        await asyncio.shield(
                            client.delete(self.base_url + f"/v1/videos/{job_id}",
                                          timeout=5)
                        )
                    except Exception:
                        pass
