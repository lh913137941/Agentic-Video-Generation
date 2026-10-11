"""Visual timeline and local marked-region editing backed by FFmpeg.

Time selection and drawing overlays are separate from V2V/Omni inference.
A marked rectangle can be blurred locally in the chosen time interval; other
natural-language edits require a capable model and are NOT faked here.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from .trim import inspect_source, trim_source_video


class MarkupError(ValueError):
    """Invalid or empty drawing region."""


def timeline_seconds(duration: float, count: int = 8) -> list[float]:
    """Endpoints included so each video boundary has a corresponding frame."""
    if duration <= 0 or not 2 <= count <= 12:
        raise ValueError("Invalid filmstrip request")
    # The container duration may exceed the last decodable video frame.
    # Leave a small safety margin rather than seeking past the final PTS.
    end = max(0.0, duration - min(0.25, max(0.10, duration*0.10)))
    return [round(end * n / (count - 1), 2) for n in range(count)]


def frame_at(path: str | Path, seconds: float, *, edge: int = 720):
    """Extract one exact-time preview frame as PIL Image (never a file URL)."""
    from PIL import Image

    file = Path(path)
    if not file.is_file() or file.suffix.lower() != ".mp4":
        raise ValueError("请先上传 MP4")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("无效的视频时间")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seconds:.3f}", "-i", str(file),
        "-vf", f"scale={edge}:{edge}:force_original_aspect_ratio=decrease:"
               "force_divisible_by=2",
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("无法提取时间点画面，请检查 FFmpeg") from exc
    if proc.returncode or not proc.stdout.startswith(b"\xff\xd8"):
        raise ValueError("无法提取时间点画面；可尝试稍微调整时间")
    with Image.open(io.BytesIO(proc.stdout)) as image:
        return image.convert("RGB").copy()


def make_filmstrip(path: str | Path, count: int = 8):
    duration = inspect_source(path)
    moments = timeline_seconds(duration, count)
    return [(frame_at(path, moment, edge=300), f"{moment:.1f}s")
            for moment in moments]


def _read_layer(layer):
    from PIL import Image
    if isinstance(layer, Image.Image):
        return layer.convert("RGBA")
    with Image.open(layer) as opened:
        return opened.convert("RGBA").copy()


def annotation_bbox(editor: dict) -> tuple[float, float, float, float]:
    """Return normalized bounds of drawn (nontransparent) editor layers.

    Only brush/shape layers are inspected; background/composite are excluded so
    an unmarked frame never accidentally selects the whole image.
    """
    from PIL import Image, ImageChops

    if not isinstance(editor, dict) or not editor.get("layers"):
        raise MarkupError("请先在画面上使用画笔标记需要编辑的区域")
    background = editor.get("background")
    if background is None:
        raise MarkupError("请先加载待标记的画面")
    base = _read_layer(background)
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    for image in editor["layers"]:
        layer = _read_layer(image)
        if layer.size != (w, h):
            # Gradio may store brush layers in editor canvas coordinates.
            # Resample alpha to the displayed background frame dimensions.
            layer = layer.resize((w, h), Image.Resampling.NEAREST)
        mask = ImageChops.lighter(mask, layer.getchannel("A"))
    rect = mask.getbbox()
    if rect is None:
        raise MarkupError("尚未画出有效区域，请用画笔在画面上涂抹")
    x0, y0, x1, y1 = rect
    if (x1 - x0) < 3 or (y1 - y0) < 3:
        raise MarkupError("标记区域太小，请扩大画笔范围")
    if (x1-x0)*(y1-y0) >= w*h*0.96:
        raise MarkupError("标记覆盖几乎整张画面；请缩小标记范围")
    return (x0/w, y0/h, x1/w, y1/h)


def store_markup(editor: dict, source: str, at: float, prompt: str):
    """Save a portable markup preview and JSON metadata without editing video."""
    from PIL import Image

    bounds = annotation_bbox(editor)
    if not source or not Path(source).is_file():
        raise MarkupError("原始素材不存在，请重新上传")
    if not 0 <= float(at) <= inspect_source(source):
        raise MarkupError("标记时间超出视频范围")
    if not prompt.strip():
        raise MarkupError("请描述标记区域需要如何修改")
    if len(prompt) > 1500:
        raise MarkupError("修改说明不得超过 1500 字")
    folder = Path(tempfile.mkdtemp(prefix="avg-markup-"))
    image = _read_layer(editor["background"])
    for entry in editor["layers"]:
        layer = _read_layer(entry)
        if layer.size != image.size:
            layer = layer.resize(image.size, Image.Resampling.NEAREST)
        image.alpha_composite(layer)
    marked = folder / "marked-frame.png"
    image.convert("RGB").save(marked, "PNG")
    doc = {
        "frame_second": round(float(at), 2),
        "normalized_bbox": [round(x, 4) for x in bounds],
        "instruction": prompt.strip(),
        "coordinate_system": "normalized_xyxy",
        "model_mask_consumed": False,
        "source_name": Path(source).name,
    }
    manifest = folder / "annotation.json"
    manifest.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return str(marked), str(manifest), bounds


def local_blur_region(
    source: str, editor: dict, start: float, end: float,
) -> tuple[str, tuple[float, float, float, float]]:
    """Produce an actual MP4 with a *stationary* annotated bounding-box blur.

    A single-frame mark isn't object tracking. We make this explicit in the
    UI; spatial/temporal mask conditioned AI editing remains unimplemented.
    """
    rect = annotation_bbox(editor)
    duration = inspect_source(source)
    if not (math.isfinite(start) and math.isfinite(end) and
            0 <= start < end <= duration + 0.05):
        raise ValueError("模糊生效区间必须落在视频时长范围内")
    last = min(end, duration)
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(source)],
        capture_output=True, text=True, timeout=25, check=False,
    )
    if info.returncode:
        raise ValueError("无法读取视频尺寸")
    try:
        dims = json.loads(info.stdout)["streams"][0]
        width, height = int(dims["width"]), int(dims["height"])
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise ValueError("视频尺寸异常") from exc

    # Expand slightly to protect strokes at bounding edges.
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, x0-0.01), max(0, y0-0.01)
    x1, y1 = min(1, x1+0.01), min(1, y1+0.01)
    left = min(max(0, int(x0*width)//2*2), width-2)
    top = min(max(0, int(y0*height)//2*2), height-2)
    right = min(width, max(left+2, int(math.ceil(x1*width/2)*2)))
    bottom = min(height, max(top+2, int(math.ceil(y1*height/2)*2)))
    cw, ch = right-left, bottom-top
    directory = Path(tempfile.mkdtemp(prefix="avg-local-edit-"))
    output = directory / "blurred.mp4"
    radius = max(1, min(12, cw//6, ch//6))
    filtergraph = (
        f"[0:v]split[base][cut];"
        f"[cut]crop={cw}:{ch}:{left}:{top},"
        f"boxblur={radius}:2[blur];"
        f"[base][blur]overlay={left}:{top}:"
        f"enable='between(t,{start:.3f},{last:.3f})'[v]"
    )
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-filter_complex", filtergraph,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2",
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=240, check=False)
        if proc.returncode or not output.exists():
            raise RuntimeError(
                "局部模糊失败：" + proc.stderr.decode(errors="replace")[-350:]
            )
        inspect_source(output)
        return str(output), rect
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
