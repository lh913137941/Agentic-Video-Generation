"""Strict FFmpeg-only P0 renderer. Never runs shell or trusts model-supplied paths."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .edit_plan import EditPlan

MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_MS = 2 * 60 * 60 * 1000


def probe_media(path: Path) -> tuple[int, bool, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of", "json", str(path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if p.returncode:
            raise ValueError("ffprobe failed: " + p.stderr[-300:])
        obj = json.loads(p.stdout)
        streams = obj.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        duration_ms = round(float(obj["format"]["duration"]) * 1000)
        if video is None or not 0 < duration_ms <= MAX_SOURCE_MS:
            raise ValueError("Expected a playable video shorter than two hours")
        return duration_ms, any(s.get("codec_type") == "audio" for s in streams), int(video["width"]), int(video["height"])
    except (OSError, subprocess.TimeoutExpired, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not inspect video with ffprobe") from exc


def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 600) -> None:
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FFmpeg unavailable or timed out") from exc
    if p.returncode:
        raise RuntimeError("FFmpeg failed: " + p.stderr[-800:])


def _srt_time(ms: int) -> str:
    hours, remain = divmod(ms, 3600000)
    minutes, remain = divmod(remain, 60000)
    seconds, millis = divmod(remain, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _caption_srt(plan: EditPlan) -> str:
    rows = []
    for i, caption in enumerate(plan.captions, 1):
        # SRT formatting: do not allow untrusted multiline markup or control chars.
        safe = " ".join(caption.text.split()).replace("<", "＜").replace(">", "＞")
        rows.append(f"{i}\n{_srt_time(caption.start_ms)} --> {_srt_time(caption.end_ms)}\n{safe}\n")
    return "\n".join(rows) + "\n"


def render_edit_plan(
    source: Path, plan: EditPlan, destination: Path,
    *, short_edge: int = 1080, fps: int = 30,
) -> dict:
    if short_edge < 144 or short_edge % 2 or not 1 <= fps <= 60:
        raise ValueError("Invalid render dimensions or frame rate")
    if not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Source missing or too large")
    if source.suffix.lower() != ".mp4":
        raise ValueError("Only MP4 source files supported")
    duration_ms, has_audio, _, _ = probe_media(source)
    plan.validate_for_source(duration_ms)
    if plan.approval != "approved":
        raise ValueError("Human approval required before rendering")
    w, h = {
        "9:16": (short_edge, short_edge * 16 // 9),
        "16:9": (short_edge * 16 // 9, short_edge),
        "1:1": (short_edge, short_edge),
    }[plan.aspect]
    # Round up to even dimensions required for yuv420p.
    w += w % 2
    h += h % 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p0-render-", dir=destination.parent) as scratch:
        folder = Path(scratch)
        manifest = []
        for i, clip in enumerate(plan.clips):
            name = f"clip-{i:03}.mp4"
            elapsed = (clip.out_ms - clip.in_ms) / 1000
            argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(source.resolve())]
            if not has_audio:
                argv += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            argv += [
                "-ss", f"{clip.in_ms / 1000:.3f}", "-t", f"{elapsed:.3f}",
                "-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
                "-vf", (f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,"
                        f"crop={w}:{h},setsar=1,format=yuv420p"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-threads", "2",
                "-movflags", "+faststart", str(folder / name),
            ]
            _run(argv, timeout=600)
            manifest.append(f"file '{name}'")
        (folder / "concat.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
        temp_out = folder / "output.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "concat", "-safe", "1", "-i", "concat.txt"]
        if plan.captions:
            (folder / "subs.srt").write_text(_caption_srt(plan), encoding="utf-8")
            command += ["-vf", "subtitles=subs.srt", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-threads", "2"]
        else:
            command += ["-c:v", "copy"]
        command += ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac",
                    "-movflags", "+faststart", str(temp_out)]
        _run(command, cwd=folder, timeout=900)
        rendered_ms, rendered_audio, rw, rh = probe_media(temp_out)
        allowed_drift = max(700, round(plan.duration_ms * .08))
        if (not rendered_audio or (rw, rh) != (w, h)
                or abs(rendered_ms - plan.duration_ms) > allowed_drift):
            raise RuntimeError("Rendered MP4 failed audio, geometry or duration verification")
        if temp_out.stat().st_size < 1024:
            raise RuntimeError("Rendered MP4 is empty")
        os.replace(temp_out, destination)
    return {
        "duration_ms": rendered_ms, "width": rw, "height": rh,
        "has_audio": rendered_audio, "source_has_audio": has_audio,
        "clips": len(plan.clips), "captions": len(plan.captions),
    }
