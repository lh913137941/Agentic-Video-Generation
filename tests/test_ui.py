"""UI wiring smoke test and prompt templates."""
import pytest

from agentic_video.ui import (
    EDIT_PRESETS, choose_edit_preset, video_edit_instruction,
    video_edit_availability, selected_preset_cards, PRESET_CARDS,
)


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
        components = demo.config["components"]
        cards = [
            component for component in components
            if component["type"] == "button"
            and "studio-preset-card" in component["props"].get("elem_classes", [])
        ]
        assert len(cards) == 4
        labels = [c["props"].get("value", "") for c in cards]
        assert any("光影氛围" in label for label in labels)
        assert any("替换物体" in label for label in labels)
        assert any("清理画面" in label for label in labels)
        assert any("镜头运动" in label for label in labels)
        assert not any(
            c["type"] == "dropdown" and c["props"].get("label") == "编辑指令模板"
            for c in components
        )
    finally:
        demo.close()



def test_edit_submit_disabled_in_mock_mode():
    notice, enabled = video_edit_availability({
        "video_edit_upload": False,
        "video_edit_backend": "unavailable-in-mock",
    })
    assert enabled is False
    assert "Mock" in notice
    assert "剧本成片" in notice


def test_configured_omni_is_not_misrepresented_as_verified_v2v():
    notice, enabled = video_edit_availability({
        "video_edit_upload": True,
        "model_v2v_support_verified": False,
    })
    assert enabled is True
    assert "不代表实际模型支持 V2V" in notice



def test_ui_template_preserves_real_prompt_and_escapes_status_errors():
    from agentic_video.ui import _job_status, _panel_head, _empty_preview
    assert "黄昏" in choose_edit_preset("白天变黄昏 · 推荐首测")
    assert "&lt;script&gt;" in _panel_head("01", "<script>alert(1)</script>")
    assert "&lt;img&gt;" in _empty_preview("<img>", "preview")
    assert "&lt;script&gt;" in _job_status({
        "status": "FAILED", "job_id": "abc", "error": "<script>"
    })


def test_preset_selection_highlights_only_selected_card():
    first = PRESET_CARDS[0][0]
    second = PRESET_CARDS[1][0]
    first_prompt, first_caption, first_states = selected_preset_cards(first)
    next_prompt, next_caption, next_states = selected_preset_cards(second)
    assert first_prompt == choose_edit_preset(first)
    assert next_prompt == choose_edit_preset(second)
    assert first_states == (True, False, False, False)
    assert next_states == (False, True, False, False)
    assert "已选择" in first_caption and "已选择" in next_caption
    with pytest.raises(ValueError):
        selected_preset_cards("not available")
