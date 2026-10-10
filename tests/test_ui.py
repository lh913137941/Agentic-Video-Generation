"""Unified composer smoke tests and untrusted text escaping."""
import pytest

from agentic_video.ui import (
    EDIT_PRESETS, PRESET_CARDS, _timeline_html, _empty_preview,
    _status_chip, _submit_button, choose_edit_preset,
    video_edit_availability, video_edit_instruction,
)
from agentic_video.workspace import EDIT, STORY, initial_workspace, choose_mode


def test_prompt_templates():
    assert len(PRESET_CARDS) == 4
    assert "黄昏" in choose_edit_preset(PRESET_CARDS[0][0])
    assert choose_edit_preset("unknown") == ""
    assert video_edit_instruction("  编辑颜色  ") == "编辑颜色"
    with pytest.raises(ValueError):
        video_edit_instruction(" ")
    with pytest.raises(ValueError):
        video_edit_instruction("s" * 3001)


def test_frontend_capabilities_fail_closed():
    notice, enabled = video_edit_availability({"video_edit_upload": False})
    assert not enabled and "Mock" in notice
    note, enabled = video_edit_availability({
        "video_edit_upload": True, "model_v2v_support_verified": False
    })
    assert enabled and "不代表" in note
    pytest.importorskip("gradio")
    disconnected = {"connected": False, "video_edit_upload": False}
    mock = {"connected": True, "video_edit_upload": False}
    assert _submit_button(initial_workspace(), disconnected)["interactive"] is False
    assert _submit_button(initial_workspace(), mock)["interactive"] is True
    editing = choose_mode(initial_workspace(), EDIT)
    assert _submit_button(editing, mock)["interactive"] is False
    assert _submit_button(editing, {
        "connected": True, "video_edit_upload": True,
    })["interactive"] is True


def test_untrusted_timeline_and_status_escaped():
    view = _timeline_html([
        {"role": "user", "message": "<script>alert(1)</script>"}
    ], STORY)
    assert "&lt;script&gt;" in view and "<script>" not in view
    assert "&lt;img&gt;" in _empty_preview("<img>", "preview")
    assert "&lt;script&gt;" in _status_chip({
        "status": "FAILED", "error": "<script>",
    })


def test_gradio_has_one_creator_no_business_tabs():
    pytest.importorskip("gradio")
    from agentic_video.ui import build_ui
    demo = build_ui("http://127.0.0.1:8000")
    try:
        components = demo.config["components"]
        types = [c["type"] for c in components]
        assert "tabs" not in types and "tabitem" not in types
        upload_videos = [
            c for c in components
            if c["type"] == "video"
            and c["props"].get("label") == "上传参考视频 · MP4"
        ]
        assert len(upload_videos) == 1
        assert upload_videos[0]["props"]["sources"] == ["upload"]
        assert upload_videos[0]["props"]["format"] == "mp4"
        assert upload_videos[0]["props"].get("height") is None
        assert upload_videos[0]["props"]["interactive"] is True
        assert not any(
            c["type"] == "file" and "上传视频" in c["props"].get("label", "")
            for c in components
        )
        prompt_fields = [
            c for c in components
            if c["type"] == "textbox"
            and c["props"].get("label") == "你的创作想法"
        ]
        assert len(prompt_fields) == 1
        texts = [
            c["props"].get("value", "") or ""
            for c in components if c["type"] == "button"
        ]
        assert "✦ 视频编辑" in texts
        assert "▦ 创作分镜" in texts
        assert sum(
            any(label in value for _, label in PRESET_CARDS)
            for value in texts
        ) == 4
        assert any("生成" in value for value in texts)
    finally:
        demo.close()


def test_video_upload_css_does_not_override_filename_contrast():
    from agentic_video.ui import STYLE_PATH
    stylesheet = STYLE_PATH.read_text(encoding="utf-8")
    assert ".studio-video-upload video" in stylesheet
    assert ".studio-compact-upload *" not in stylesheet
    assert "background:#fff!important" in stylesheet


def test_video_editor_styles_allow_trim_controls():
    from agentic_video.ui import STYLE_PATH
    css = STYLE_PATH.read_text(encoding="utf-8")
    block = css.split(".studio-video-upload {", 1)[1].split("}", 1)[0]
    assert "min-height:318px" in block
    assert "overflow:visible!important" in block
    assert "overflow:hidden" not in block
    assert "height:225px!important" in css
    assert ".studio-composer-card {overflow:visible!important;}" in css
