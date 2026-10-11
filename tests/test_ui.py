"""Unified composer smoke tests and untrusted text escaping."""
import pytest

from agentic_video.ui import (
    EDIT_PRESETS, PRESET_CARDS, _timeline_html, _analysis_html, _empty_preview,
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
        uploads = [
            c for c in components
            if c["type"] == "file"
            and c["props"].get("label") == "上传参考视频 · MP4"
        ]
        assert len(uploads) == 1
        assert uploads[0]["props"]["file_types"] == [".mp4"]
        assert uploads[0]["props"]["file_count"] == "single"
        previews = [
            c for c in components
            if c["type"] == "video"
            and c["props"].get("label") == "素材预览"
        ]
        assert len(previews) == 1
        assert previews[0]["props"]["interactive"] is False
        assert previews[0]["props"]["height"] == 208
        assert not any(
            c["type"] == "video" and c["props"].get("interactive") is True
            for c in components
        )
        labels = [c["props"].get("label") for c in components]
        assert "裁剪起点 · 秒" in labels
        assert "裁剪终点 · 秒" in labels
        assert "起点画面 · START" in labels
        assert "终点画面 · END" in labels
        assert "画面标记画板 · 用画笔涂抹目标区域" in labels
        assert "区域编辑说明 / Prompt" in labels
        assert "下载已裁剪 / 已模糊的本地 MP4" in labels
        assert sum(c["type"] == "imageeditor" for c in components) == 1
        assert sum(c["type"] == "gallery" for c in components) >= 1
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
        assert "◉ 视频理解" in texts
        assert sum(
            any(label in value for _, label in PRESET_CARDS)
            for value in texts
        ) == 4
        assert any("生成" in value for value in texts)
    finally:
        demo.close()


def test_separate_file_preview_trim_styles():
    from agentic_video.ui import STYLE_PATH
    css = STYLE_PATH.read_text(encoding="utf-8")
    assert ".studio-file-upload" in css
    assert ".studio-video-preview video" in css
    assert "height:208px!important" in css
    assert ".studio-trim-tools" in css
    assert ".studio-composer-card {overflow:visible!important;}" in css
    assert ".studio-compact-upload *" not in css
    assert ".studio-video-upload" not in css


def test_upload_error_displayed_outside_collapsed_trim_controls():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    assert 'warning = str(exc)' in source
    assert 'gr.update(value=uploaded if valid else None)' in source
    assert 'uploaded if valid else None' in source
    assert 'Keep errors visible even while the trim panel is closed.' in source


def test_multimodal_model_output_safely_rendered():
    rendered = _analysis_html({
        "summary": "<script>alert(1)</script>",
        "observations": ["<b>bold?</b>"],
        "sampled_seconds": [0.6, 2.2],
    })
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;b&gt;" in rendered
    assert "0.6s" in rendered


def test_timeline_markup_controls_have_real_action_events():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    assert "frame_strip.select(" in source
    assert "trim_start.release(" in source
    assert "trim_end.release(" in source
    assert "mark_load.click(" in source
    assert "save_mark.click(" in source
    assert "blur_button.click(" in source
    assert "local_blur_region(" in source
