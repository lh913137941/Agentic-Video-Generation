"""Video keyframes -> OpenAI-compatible vision API -> structured storyboard.

Only sampled JPEGs and the instruction leave the local server. Audio is NOT read.
"""
import asyncio
import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi import UploadFile
from pydantic import BaseModel, Field
from .models import Project, ShotSpec
from .trim import inspect_source


class VisionError(RuntimeError):
    """Sanitized provider / model error shown to clients."""


class VisionPlan(BaseModel):
    summary: str = Field(min_length=1, max_length=1500)
    observations: list[str] = Field(default_factory=list, max_length=12)
    shots: list[ShotSpec] = Field(min_length=1, max_length=10)


class VideoUnderstandingResult(BaseModel):
    summary: str
    observations: list[str]
    sampled_seconds: list[float]
    project: Project


@dataclass(frozen=True)
class VisionConfig:
    base_url: str
    model: str
    api_key: str = ""
    max_frames: int = 4
    frame_edge: int = 512
    timeout_seconds: int = 120

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("AVG_VISION_MODEL required")
        if not (1 <= self.max_frames <= 8 and 128 <= self.frame_edge <= 1024):
            raise ValueError("Invalid vision frames or resolution")
        if not (10 <= self.timeout_seconds <= 600):
            raise ValueError("Invalid vision timeout")
        url = httpx.URL(self.base_url)
        if not url.host or url.query or url.fragment or url.userinfo:
            raise ValueError("Invalid vision base URL")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.host in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Vision API must use HTTPS except localhost")


def sample_seconds(duration: float, count: int) -> list[float]:
    return [
        round(max(0.01, min(duration - 0.05, (i + 0.5) * duration / count)), 2)
        for i in range(count)
    ]


def extract_frame(path: Path, second: float, edge: int) -> bytes:
    try:
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{second:.3f}", "-i", str(path),
            "-vf", (f"scale={edge}:{edge}:force_original_aspect_ratio=decrease:"
                    "force_divisible_by=2"),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ], capture_output=True, timeout=25, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VisionError("FFmpeg 关键帧提取失败") from exc
    if result.returncode or not result.stdout.startswith(b"\xff\xd8"):
        raise VisionError("视频无法解码为关键帧")
    if len(result.stdout) > 2 * 1024 * 1024:
        raise VisionError("关键帧过大")
    return result.stdout


class OpenAICompatibleVisionAnalyzer:
    def __init__(self, config: VisionConfig):
        self.config = config
        root = config.base_url.rstrip("/")
        self.url = root + (
            "/chat/completions" if root.endswith("/v1")
            else "/v1/chat/completions"
        )

    async def analyze(
        self, upload: UploadFile, instruction: str,
        shot_count: int, style: str,
    ) -> tuple[VisionPlan, list[float]]:
        if not 1 <= shot_count <= 10:
            raise ValueError("shot_count must be 1..10")
        if not instruction.strip() or len(instruction) > 3000 or len(style) > 500:
            raise ValueError("Invalid instruction or style")
        if not upload.filename or Path(upload.filename).suffix.lower() != ".mp4":
            raise ValueError("仅支持 MP4 上传")
        with TemporaryDirectory(prefix="avg-vision-") as folder:
            path = Path(folder) / "source.mp4"
            try:
                count = 0
                with path.open("wb") as out:
                    while block := await upload.read(1024 * 1024):
                        count += len(block)
                        if count > 100 * 1024 * 1024:
                            raise ValueError("视频不得超过 100 MiB")
                        out.write(block)
                duration = await asyncio.to_thread(inspect_source, path)
                times = sample_seconds(duration, self.config.max_frames)
                frames = await asyncio.gather(*[
                    asyncio.to_thread(
                        extract_frame, path, t, self.config.frame_edge
                    ) for t in times
                ])
            finally:
                await upload.close()
            content = [{
                "type": "text",
                "text": (
                    f"请基于这些抽样视频帧理解可见内容，用户目标：{instruction}。"
                    f"请按照 {style} 风格规划恰好 {shot_count} 个镜头。"
                    "只输出 JSON 对象：summary（概述）、observations（可见事实数组）、"
                    "shots（对象数组，含 prompt、duration、camera_motion）。"
                    "duration 限 1～10 秒。不得声称听到音频或看到未采样画面。"
                )
            }]
            for sec, frame in zip(times, frames):
                content.extend([
                    {"type": "text", "text": f"时间戳 {sec:.2f} 秒"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," +
                               base64.b64encode(frame).decode("ascii"),
                        "detail": "low",
                    }},
                ])
            payload = {
                "model": self.config.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": (
                        "Analyze only visual evidence. Return strict JSON, no Markdown."
                    )},
                    {"role": "user", "content": content},
                ],
            }
            headers = (
                {"Authorization": f"Bearer {self.config.api_key}"}
                if self.config.api_key else {}
            )
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.timeout_seconds, follow_redirects=False
                ) as client:
                    response = await client.post(
                        self.url, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                if not isinstance(raw, str):
                    raise ValueError("No text response")
                plan = VisionPlan.model_validate(json.loads(raw))
                if len(plan.shots) != shot_count:
                    raise ValueError("Wrong number of shots")
                return plan, times
            except httpx.HTTPStatusError as exc:
                raise VisionError(
                    f"视觉模型 API 返回 HTTP {exc.response.status_code}，请检查模型权限"
                ) from None
            except httpx.HTTPError:
                raise VisionError("视觉模型 API 连接失败或超时") from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise VisionError("视觉模型未返回符合 JSON 分镜约定的内容") from None
