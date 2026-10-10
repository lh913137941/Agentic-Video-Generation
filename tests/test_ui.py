"""UI wiring smoke test and prompt templates."""
import pytest

from agentic_video.ui import EDIT_PRESETS, choose_edit_preset, video_edit_instruction


def test_prompt_templates_are_explicit_video_edits():
    assert "黄昏" in choose_edit_preset("白天变黄昏 · 推荐首测")
    assert len(EDIT_PRESETS) >= 3
    assert choose_edit_preset("unknown") == ""
    assert video_edit_instruction("  编辑颜色  ") == "编辑颜色"
    with pytest.raises(ValueError):
        video_edit_instruction("   ")
    with pytest.raises(ValueError):
        video_edit_instruction("s" * 3001)


def test_gradio_components_construct():
    pytest.importorskip("gradio")
    from agentic_video.ui import build_ui
    demo = build_ui("http://127.0.0.1:8000")
    try:
        assert demo is not None
    finally:
        demo.close()
