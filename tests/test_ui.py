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
        assert "直接在画面上标记目标" in labels
        assert "描述如何调整画面" in labels
        assert "下载处理后的 MP4" in labels
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


def test_jimeng_inspired_focus_mode_has_real_open_save_close_events():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    assert 'elem_id="studio-edit-overlay"' in source
    assert 'visible=False, elem_id="studio-edit-overlay"' in source
    assert 'advanced_button.click(' in source
    assert 'close_editor_button.click(' in source
    assert 'fn=save_advanced_editor' in source
    assert 'outputs=[\n                edit_overlay, prompt, marked_preview, manifest_file, mark_status,' in source
    assert 'gr.ImageEditor(' in source
    assert '精确裁剪与本地处理（可选）' in source
    assert 'with gr.Accordion(\n                "✂ 时间轴裁剪与画面标记' not in source


def test_immersive_editor_has_dark_style_and_not_a_fake_html_button():
    from agentic_video.ui import STYLE_PATH
    css = STYLE_PATH.read_text(encoding="utf-8")
    assert "#studio-edit-overlay {" in css
    assert "position:fixed!important" in css
    assert ".studio-edit-command" in css
    assert ".studio-edit-save" in css
    assert ".studio-composer-card .studio-main-prompt textarea" in css
    assert "background:#111214!important" in css


def test_jimeng_style_direct_frame_navigation():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    assert "editor_frames = gr.Gallery(" in source
    assert "editor_frames.select(" in source
    assert "fn=choose_editor_frame" in source
    assert "outputs=[marker, mark_second, mark_status]" in source
    assert "studio-edit-command" in source
    assert "精确裁剪与本地处理（可选）" in source


def test_marked_frame_visible_in_composer_after_save():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    assert 'as annotation_card:' in source
    assert 'label="已保存的标记画面"' in source
    assert 'record_annotation(' in source
    assert '还没有画面标记' in source
    assert 'annotation_card, annotation_thumb, workspace, annotation_note' in source
    assert 'outputs=[annotation_card, annotation_thumb]' in source
    css = ui.STYLE_PATH.read_text(encoding="utf-8")
    assert '#studio-edit-overlay .studio-edit-command {' in css
    assert 'position:static!important;' in css
    assert '.studio-annotation-card {' in css


def test_viewport_adaptive_editor_and_explicit_v2v_configuration():
    import inspect
    from agentic_video import ui
    css = ui.STYLE_PATH.read_text(encoding="utf-8")
    assert 'height:calc(100dvh - 142px)!important' in css
    assert 'min-height:0!important' in css
    assert 'overflow:hidden!important' in css
    assert 'height=600, type="pil"' in inspect.getsource(ui.build_ui)
    assert "精确定位画面 · 可选" in inspect.getsource(ui.build_ui)
    src = inspect.getsource(ui.build_ui)
    assert "⚙ V2V 模型 API 配置" in src
    assert 'type="password"' in src
    assert "save_local_v2v(" in src
    assert "save_provider.click(" in src
    assert "detect_provider.click(" in src


def test_advanced_editor_uses_two_panes_and_no_legacy_stack():
    import inspect
    from agentic_video import ui
    source = inspect.getsource(ui.build_ui)
    css = ui.STYLE_PATH.read_text(encoding="utf-8")
    assert 'elem_classes=["studio-edit-body"]' in source
    assert 'elem_classes=["studio-edit-canvas-pane"]' in source
    assert 'elem_classes=["studio-edit-inspector"]' in source
    assert 'canvas_size=(960, 540)' in source
    assert 'columns=4, rows=2, height=165' in source
    assert 'elem_classes=["studio-edit-command"]' in source
    assert 'save_mark.click(' in source
    assert 'editor_frames.select(' in source
    # Structural regression: no more multiple overlapping CSS generations.
    assert "690px!important" not in css
    assert "47dvh" not in css
    assert "position:sticky!important" not in css
    assert css.count("/* Advanced editor v3:") == 1
    assert "overflow-y:auto!important" in css
    assert "flex-direction:row!important" in css


def test_advanced_editor_config_preserves_real_drawing_and_frame_selection():
    import pytest
    pytest.importorskip("gradio")
    from agentic_video.ui import build_ui
    demo = build_ui("http://127.0.0.1:8000")
    try:
        components = demo.config["components"]
        editor = [c for c in components if c["type"] == "imageeditor"]
        assert len(editor) == 1
        assert editor[0]["props"]["height"] == 600
        assert editor[0]["props"]["canvas_size"] == [960, 540]
        frame = next(
            c for c in components
            if c["type"] == "gallery"
            and c["props"].get("label") == "点击缩略图切换标记画面"
        )
        assert frame["props"]["columns"] == 4
        assert frame["props"]["rows"] == 2
        assert frame["props"]["height"] == 165
        deps = demo.config["dependencies"]
        assert any(
            item.get("targets") and item.get("outputs")
            for item in deps
        )
    finally:
        demo.close()
