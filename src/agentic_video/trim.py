"""Simple accurate FFmpeg time trimming of locally uploaded MP4 clips.

Independent from the AI/V2V backend. No shell, URLs or model calls.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .editor import probe_video

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_CLIP_SECONDS = 30


class MediaToolsUnavailable(ValueError):
    """A required executable is not accessible in the current Python PATH."""


def check_media_tools() -> None:
    missing = [
        tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)
    ]
    if missing:
        names = "/".join(missing)
        raise MediaToolsUnavailable(
            f"本机找不到 {names}。Windows 请在 PowerShell 执行 "
            "`winget install --id Gyan.FFmpeg --exact`，"
            "关闭所有终端并重新打开 Git Bash，然后检查 "
            "`ffmpeg -version` 和 `ffprobe -version`，最后重启 Gradio。"
        )


def inspect_source(path: str | Path) -> float:
    """Validate a Gradio-uploaded MP4 and report its duration."""
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("请先上传有效的 MP4 文件")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("视频文件不可超过 100 MiB")
    check_media_tools()
    try:
        return probe_video(source)
    except FileNotFoundError as exc:
        raise MediaToolsUnavailable(
            "ffprobe 无法启动。请确认 FFmpeg 的 bin 目录在 PATH 中，"
            "然后重新打开 Git Bash 并重启 Gradio。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("视频分析超时，请尝试较短的 MP4 片段") from exc
    except OSError as exc:
        raise ValueError("无法读取视频文件，请检查权限和路径") from exc


def trim_source_video(path: str | Path, start: float, end: float) -> str:
    """Return a new MP4 between two seconds; do not modify the uploaded source."""
    source = Path(path)
    duration = inspect_source(source)
    try:
        first, last = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("请指定有效的开始/结束秒数") from exc
    if not (0 <= first < last <= duration + 0.05 and (last - first) >= 0.2):
        raise ValueError(f"裁剪时间必须在 0～{duration:.2f} 秒内且片段至少 0.2 秒")
    last = min(last, duration)
    target = tempfile.mkdtemp(prefix="avg-clip-")
    result = Path(target) / "trimmed.mp4"
    # Re-encode for frame-accurate cuts, keep audio when present.
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{first:.3f}", "-i", str(source),
        "-t", f"{last - first:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2",
        "-c:a", "aac", "-movflags", "+faststart", str(result),
    ]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=240, check=False)
        if proc.returncode:
            raise RuntimeError(
                "裁剪失败，FFmpeg: " +
                proc.stderr.decode(errors="replace")[-500:]
            )
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError("裁剪后未产生有效 MP4")
        inspect_source(result)
        return str(result)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError("视频裁剪超时或 FFmpeg 不可用") from exc
    finally:
        if not result.is_file() or not result.stat().st_size:
            shutil.rmtree(target, ignore_errors=True)
