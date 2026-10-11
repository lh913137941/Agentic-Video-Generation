"""Visual frame trim, hand-painted region markup and REAL local MP4 edits."""
import asyncio
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from agentic_video.visual_editor import (
    MarkupError, annotation_bbox, frame_at, local_blur_region,
    make_filmstrip, store_markup, timeline_seconds,
)
from agentic_video.trim import inspect_source


def drawing():
    background = Image.new("RGB", (240, 140), "#ddddee")
    paint = Image.new("RGBA", (240, 140), (0, 0, 0, 0))
    ImageDraw.Draw(paint).rectangle((45, 30, 110, 90),
                                    fill=(255, 30, 40, 180))
    return {"background": background, "layers": [paint],
            "composite": background}


def test_frame_sampling_boundaries_are_observable():
    values = timeline_seconds(5.09, 8)
    assert len(values) == 8
    assert values[0] == 0.0
    assert 4.9 <= values[-1] <= 5.09
    assert sorted(values) == values
    with pytest.raises(ValueError):
        timeline_seconds(-1)


def test_editor_uses_drawn_alpha_not_full_background():
    box = annotation_bbox(drawing())
    assert 0.15 < box[0] < 0.25
    assert 0.7 < box[2] - box[0] + 0.5 < 1.0
    with pytest.raises(MarkupError):
        annotation_bbox({"background": Image.new("RGB", (200, 200)),
                         "layers": [], "composite": None})


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Requires FFmpeg",
)
def test_frame_preview_markup_export_and_local_blur(tmp_path):
    from agentic_video.backends import MockVideoBackend

    original = tmp_path / "clip.mp4"
    asyncio.run(MockVideoBackend(fps=12, width=240, height=140).generate(
        "street trees moving", original, 2.5
    ))
    before_bytes = original.read_bytes()
    duration = inspect_source(original)
    assert duration > 2
    start_frame = frame_at(original, 0)
    end_frame = frame_at(original, duration-0.1)
    assert start_frame.size[0] >= 200
    assert end_frame.size == start_frame.size
    filmstrip = make_filmstrip(original)
    assert len(filmstrip) == 8
    assert filmstrip[0][1] == "0.0s"

    image, metadata, box = store_markup(
        drawing(), str(original), 1.0, "将框选区域中的物体模糊"
    )
    assert Path(image).is_file() and Path(metadata).is_file()
    parsed = json.loads(Path(metadata).read_text(encoding="utf-8"))
    assert parsed["model_mask_consumed"] is False
    assert parsed["instruction"] == "将框选区域中的物体模糊"
    assert parsed["normalized_bbox"] == [round(n, 4) for n in box]

    edited, rect = local_blur_region(
        str(original), drawing(), 0.5, 1.9
    )
    assert Path(edited).is_file()
    assert abs(inspect_source(edited) - duration) < 0.2
    assert Path(edited).read_bytes() != before_bytes
    assert original.read_bytes() == before_bytes
    assert rect == box

    with pytest.raises(ValueError, match="生效区间"):
        local_blur_region(str(original), drawing(), 1.9, 0.4)


def test_invalid_markup_rejects_unmarked_layer(tmp_path):
    empty = {
        "background": Image.new("RGB", (240, 140), "blue"),
        "layers": [Image.new("RGBA", (240, 140), (0, 0, 0, 0))],
        "composite": None,
    }
    with pytest.raises(MarkupError, match="有效区域"):
        annotation_bbox(empty)
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"x")
    from unittest.mock import patch
    with patch("agentic_video.visual_editor.inspect_source", return_value=2.0):
        with pytest.raises(MarkupError, match="描述"):
            store_markup(drawing(), str(fake), 0.8, "")
