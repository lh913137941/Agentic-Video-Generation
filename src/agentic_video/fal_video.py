"""Fal Wan video-to-video editing adapter.

Explicitly NOT a generic OpenAI-compatible multimodal chat endpoint.
Requires the optional fal-client dependency and a configured fal.ai API key.
"""
import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from .backends import BackendError

MODEL = "fal-ai/wan/v2.7/edit-video"
MAX_RESULT_BYTES = 200 * 1024 * 1024


def _trusted_output_url(value: str) -> str:
    if not isinstance(value, str):
        raise BackendError("fal.ai 未返回有效视频 URL")
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "fal.media" or host.endswith(".fal.media")
    ) or parsed.username or parsed.password:
        raise BackendError("fal.ai 返回的视频地址不在允许的 CDN 域名内")
    return value


class FalWanEditBackend:
    def __init__(self, api_key: str, model: str = MODEL,
                 timeout_seconds: int = 1800):
        if model != MODEL or not api_key.strip():
            raise ValueError("需要 fal.ai Wan 2.7 Edit Video 和有效 API Key")
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def validate_duration(duration: float) -> None:
        if not 2 <= duration <= 10:
            raise ValueError("fal.ai Wan Edit Video 只接受 2～10 秒的 MP4，请先裁剪")

    async def generate(
        self, prompt: str, destination: Path, duration: float,
        input_reference: Path | None = None,
    ) -> None:
        if input_reference is None:
            raise BackendError("该 fal.ai 视频编辑模型必须提供原视频")
        if not 2 <= duration <= 10:
            raise BackendError("fal.ai Wan Edit Video 要求原片时长为 2～10 秒")
        try:
            import fal_client
        except ImportError as exc:
            raise BackendError(
                "未安装云端编辑依赖，请执行 pip install -e '.[ui,v2v]'"
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        # Auth stays server-side; do not mutate global FAL_KEY.
        client = fal_client.AsyncClient(key=self.api_key)
        try:
            video_url = await client.upload_file(str(input_reference))
            handle = await client.submit(
                self.model, arguments={
                    "prompt": prompt, "video_url": video_url,
                    "resolution": "720p", "audio_setting": "origin",
                }
            )
            result = await asyncio.wait_for(
                handle.get(), timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise BackendError("fal.ai 视频编辑超时，请稍后查看任务状态") from exc
        except Exception as exc:
            # Do not echo SDK exceptions, which can contain API credentials.
            raise BackendError(
                "fal.ai 视频编辑请求失败，请检查密钥、权限、额度及网络"
            ) from None

        media_url = _trusted_output_url((result.get("video") or {}).get("url"))
        partial = destination.with_suffix(".partial.mp4")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30, read=120),
                follow_redirects=False,
            ) as client_http:
                async with client_http.stream("GET", media_url) as response:
                    response.raise_for_status()
                    size = 0
                    with partial.open("wb") as stream:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_RESULT_BYTES:
                                raise BackendError("fal.ai 输出视频超过 200 MiB 限制")
                            stream.write(chunk)
            if not partial.is_file() or partial.stat().st_size < 12:
                raise BackendError("fal.ai 未返回有效的视频内容")
            with partial.open("rb") as data:
                if data.read(12)[4:8] != b"ftyp":
                    raise BackendError("fal.ai 输出不是 MP4 文件")
            os.replace(partial, destination)
        except httpx.HTTPError:
            raise BackendError("无法从 fal.ai 下载编辑视频，请稍后重试") from None
        finally:
            partial.unlink(missing_ok=True)
