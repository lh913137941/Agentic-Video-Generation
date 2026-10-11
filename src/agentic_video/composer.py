"""FFmpeg-based normalized concat. Re-encodes deliberately; not stream-copy."""
import os
import subprocess
from pathlib import Path
from uuid import uuid4


class FFmpegComposer:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 24,
                 timeout: int = 240):
        self.width, self.height, self.fps, self.timeout = width, height, fps, timeout

    def compose(self, clips: list[tuple[Path, float]], output: Path) -> None:
        if not clips or any(not path.is_file() for path, _ in clips):
            raise ValueError("Every storyboard clip must exist before composition")
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(f".{uuid4().hex}.mp4")
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for path, _ in clips:
            args.extend(["-i", str(path)])
        filters = []
        for index, (_, duration) in enumerate(clips):
            filters.append(
                f"[{index}:v]fps={self.fps},"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,format=yuv420p,"
                f"tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS[v{index}]"
            )
        inputs = "".join(f"[v{i}]" for i in range(len(clips)))
        filters.append(f"{inputs}concat=n={len(clips)}:v=1:a=0[outv]")
        args += [
            "-filter_complex", ";".join(filters), "-map", "[outv]",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(self.fps), "-threads", "2",
            "-movflags", "+faststart", str(tmp),
        ]
        try:
            result = subprocess.run(
                args, capture_output=True, check=False, timeout=self.timeout
            )
            if result.returncode != 0 or not tmp.is_file() or not tmp.stat().st_size:
                raise RuntimeError(
                    "FFmpeg compose failed: " +
                    result.stderr.decode(errors="replace")[-1000:]
                )
            os.replace(tmp, output)
        finally:
            tmp.unlink(missing_ok=True)
