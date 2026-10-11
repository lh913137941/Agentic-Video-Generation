"""Unified single-page creator composer and contextual project workspace.

Interactive UI controls are Gradio components. The conversation timeline is
UI feedback, not an LLM agent or persistent unified backend project.
"""
from __future__ import annotations

import html
import os
from functools import partial
from pathlib import Path

import httpx

from .trim import inspect_source, trim_source_video
from .visual_editor import (
    annotation_bbox, frame_at, local_blur_region, make_filmstrip,
    store_markup, timeline_seconds,
)
from .workspace import (
    EDIT, STORY, VISION, append_event, attach_video, begin_submission, choose_mode,
    has_active_job, initial_workspace, plan_submission, update_task,
)

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "studio.css"

EDIT_PRESETS = {
    "白天变黄昏 · 推荐首测": (
        "视频编辑：将上传视频中的白天环境修改为黄昏时分，天空变成橙红色晚霞。\n"
        "保持原视频中的人物身份、服装、动作、位置、镜头运动及视频节奏不变。\n"
        "仅修改天空、环境光照与场景色温，调整人物和物体受光，使画面自然真实。\n"
        "保持原视频的画幅比例和时长，不增加新的镜头。"
    ),
    "替换桌上马克杯": (
        "视频编辑：将画面中桌上的红色马克杯替换成透明玻璃杯，"
        "保持杯子的位置、大小、透视及环境光照不变，不修改其他物体。"
    ),
    "移除背景垃圾桶": (
        "视频编辑：移除视频背景中的垃圾桶，并根据周围墙面、"
        "地面和光照自然补全，保持其他物体和镜头运动不变。"
    ),
    "缓慢推近镜头 · 探索": (
        "视频编辑：保持所有人物、动作、服装和场景不变，"
        "将原有固定镜头改为缓慢向前推进的电影镜头，画面自然稳定。"
    ),
}
PRESET_CARDS = (
    ("白天变黄昏 · 推荐首测", "🌇 黄昏光影"),
    ("替换桌上马克杯", "☕ 替换物体"),
    ("移除背景垃圾桶", "✦ 移除物体"),
    ("缓慢推近镜头 · 探索", "◈ 镜头推进"),
)
STATUS_LABELS = {
    "QUEUED": ("排队中", "running"),
    "GENERATING": ("生成中", "running"),
    "STITCHING": ("正在合成", "running"),
    "READY": ("已完成", "ready"),
    "FAILED": ("执行失败", "failed"),
    "DRAFT": ("分镜待生成", ""),
    "PENDING": ("待生成", ""),
}


def choose_edit_preset(name: str) -> str:
    return EDIT_PRESETS.get(name, "")


def video_edit_instruction(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("请先输入编辑指令")
    if len(prompt) > 3000:
        raise ValueError("编辑指令过长（最多 3000 字符）")
    return prompt


def video_edit_availability(info: dict) -> tuple[str, bool]:
    if not info.get("video_edit_upload"):
        return (
            "当前为 Mock 模式：可体验剧本分镜，但不能对上传视频做 AI 编辑。"
            "真正视频编辑需在后端部署支持 V2V 的模型。",
            False,
        )
    return (
        "Omni 推理接口已配置，但不代表实际模型支持 V2V 的 "
        "input_reference；请先确认模型能力。",
        True,
    )


def _empty_preview(title: str, detail: str, icon: str = "▧") -> str:
    return (
        '<div class="studio-result-shell">'
        f'<div class="empty-mark">{html.escape(icon)}</div>'
        f'<strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(detail)}</span></div>'
    )


def _status_chip(item: dict | None) -> str:
    status = str((item or {}).get("status", "DRAFT"))
    label, color = STATUS_LABELS.get(status, (status, ""))
    error = html.escape(str((item or {}).get("error") or ""))
    return (
        '<div class="studio-result-meta">'
        f'<span class="studio-status-chip {color}">{html.escape(label)}</span>'
        + (f'<small>{error}</small>' if error else "")
        + '</div>'
    )


def _timeline_html(events: list[dict], mode: str) -> str:
    if not events:
        return (
            '<div class="studio-empty-conversation">'
            '<div class="studio-sparkle">✦</div>'
            '<b>创作记录将在这里出现</b>'
            '<p>上传素材或输入剧本，选择技能，然后开始创作。</p>'
            '</div>'
        )
    parts = []
    for event in events[-12:]:
        role = "user" if event.get("role") == "user" else "system"
        marker = "你" if role == "user" else "✦"
        parts.append(
            f'<div class="studio-chat-line {role}">'
            f'<div class="studio-chat-avatar">{marker}</div>'
            '<div class="studio-chat-bubble">'
            f'{html.escape(str(event.get("message", "")))}</div></div>'
        )
    return '<div class="studio-conversation">' + "".join(parts) + '</div>'


def _analysis_html(result: dict | None) -> str:
    if not result:
        return _empty_preview(
            "等待视频理解", "上传 MP4，选择视频理解后可自动生成分镜草稿"
        )
    summary = html.escape(str(result.get("summary", ""))[:1500])
    facts = "".join(
        "<li>" + html.escape(str(item)[:500]) + "</li>"
        for item in result.get("observations", [])[:12]
    )
    timestamps = ", ".join(
        str(x) + "s" for x in result.get("sampled_seconds", [])[:8]
    )
    return (
        '<div class="studio-analysis">'
        '<h3>视觉理解摘要</h3><p>' + summary + '</p>'
        '<h4>模型识别的可见内容</h4><ul>' + facts + '</ul>'
        '<p class="studio-hint">抽样时间：' + html.escape(timestamps)
        + '。只分析抽样视频帧，不解析音频。</p>'
        '</div>'
    )


def _mode_description(state: dict, capabilities: dict) -> str:
    mode = state.get("mode", STORY)
    connected = capabilities.get("connected", False)
    if not connected:
        return "尚未连接 API，请先启动后端。"
    if mode == EDIT:
        notice, _ = video_edit_availability(capabilities)
        return notice
    if mode == VISION:
        if not capabilities.get("video_understanding"):
            return "视频理解未配置。请在后端环境设置 AVG_VISION_URL、AVG_VISION_MODEL 和 AVG_VISION_API_KEY。"
        return (
            "视频理解已配置：" + str(capabilities.get("vision_model") or "多模态模型")
            + "。仅发送采样图片，模型兼容性尚未实机验证；"
            "不上传音频、不自动执行视频生成。"
        )
    if state.get("source_name"):
        return "创作分镜当前不使用视频参考，请先移除上传的 MP4 或切换视频编辑。"
    return "剧本 → 分镜 → 镜头生成 → 粗剪合成。Mock 模式也能体验完整流程。"


def _submit_button(state: dict, capabilities: dict):
    import gradio as gr
    mode = state.get("mode", STORY)
    connected = bool(capabilities.get("connected"))
    if not connected:
        return gr.update(interactive=False, value="API 未连接")
    if mode == VISION:
        if not capabilities.get("video_understanding"):
            return gr.update(interactive=False, value="视频理解 API 尚未配置")
        if not state.get("source_name"):
            return gr.update(interactive=False, value="请先上传 MP4 进行理解")
        return gr.update(interactive=True, value="◉ 理解视频并生成分镜草稿")
    if mode == EDIT:
        if not capabilities.get("video_edit_upload"):
            return gr.update(interactive=False, value="视频编辑需要 V2V 模型")
        if has_active_job(state, EDIT):
            return gr.update(interactive=False, value="视频正在生成…")
        return gr.update(interactive=True, value="✦ 生成编辑视频")
    if state.get("source_name"):
        return gr.update(interactive=False, value="请先移除视频素材")
    if has_active_job(state, STORY):
        return gr.update(interactive=False, value="正在生成分镜粗剪…")
    return gr.update(interactive=True, value="✦ 生成分镜粗剪")


def _mode_buttons(mode: str):
    import gradio as gr
    return (
        gr.update(variant="primary" if mode == EDIT else "secondary"),
        gr.update(variant="primary" if mode == STORY else "secondary"),
        gr.update(variant="primary" if mode == VISION else "secondary"),
    )


def build_ui(api_url: str | None = None):
    import gradio as gr

    base = (api_url or os.getenv("AVG_API_URL", "http://127.0.0.1:8000")).rstrip("/")

    def send(method: str, path: str, payload=None, files=None, data=None):
        try:
            with httpx.Client(timeout=180) as client:
                response = client.request(
                    method, base + path, json=payload, files=files, data=data
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
            raise gr.Error(
                f"请求失败（{exc.response.status_code}）：{detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise gr.Error(f"无法访问视频 API（{base}）：{exc}") from exc

    def detect_capabilities(state):
        try:
            with httpx.Client(timeout=8) as client:
                response = client.get(base + "/api/capabilities")
                response.raise_for_status()
                data = response.json()
                capabilities = {
                    "connected": True,
                    "video_edit_upload": bool(data.get("video_edit_upload")),
                    "video_understanding": bool(data.get("video_understanding")),
                    "vision_model": data.get("vision_model"),
                    "model_v2v_support_verified":
                        bool(data.get("model_v2v_support_verified")),
                }
        except (httpx.HTTPError, ValueError):
            capabilities = {
                "connected": False, "video_edit_upload": False,
                "video_understanding": False
            }
        return (
            capabilities, _mode_description(state, capabilities),
            _submit_button(state, capabilities),
        )

    def select_mode(state, capabilities, mode):
        result = choose_mode(state, mode)
        return (
            result,
            _mode_description(result, capabilities),
            *_mode_buttons(mode),
            _submit_button(result, capabilities),
        )

    def upload_changed(uploaded, state, capabilities):
        # A missing FFprobe is an environment issue, not a failed video edit.
        # Keep the filename attached but don't send an unverified file to the
        # preview/player or V2V backend. Show the recovery hint in the page.
        duration = None
        warning = None
        if uploaded:
            try:
                duration = inspect_source(uploaded)
            except ValueError as exc:
                warning = str(exc)
        valid = bool(uploaded) and warning is None
        result = attach_video(state, uploaded if valid else None)
        ceiling = max(0.2, round(duration or 30, 2))
        status = (
            f"⚠️ {warning}" if warning else
            (f"✓ 已加载 MP4 · {duration:.2f} 秒。"
             "可直接使用，或展开裁剪工具。" if valid
             else "尚未上传视频 · 拖入 MP4 或点击上传")
        )
        return (
            result, _mode_description(result, capabilities),
            *_mode_buttons(result["mode"]),
            _submit_button(result, capabilities),
            uploaded if valid else None,
            gr.update(value=uploaded if valid else None),
            gr.update(value=0, maximum=ceiling, interactive=valid),
            gr.update(
                value=round(duration, 2) if valid else 10,
                maximum=ceiling, interactive=valid,
            ),
            status, None,
        )

    def setup_visual_timeline(source, first, last):
        if not source:
            return [], None, None, gr.update(
                value=0, maximum=30, interactive=False
            ), None, "请先上传有效的 MP4 素材"
        try:
            duration = inspect_source(source)
            first = min(max(float(first), 0), duration)
            last = min(max(float(last), 0), duration)
            return (
                make_filmstrip(source),
                frame_at(source, first),
                frame_at(source, max(0, last - 0.05)),
                gr.update(value=first, maximum=duration, interactive=True),
                frame_at(source, first),
                "时间轴已加载。可选择关键帧设置起止位置，"
                "也可以直接在已载入的帧上进行标记。",
            )
        except (ValueError, RuntimeError) as exc:
            return [], None, None, gr.update(
                value=0, interactive=False
            ), None, f"⚠️ 时间轴初始化失败：{exc}"

    def preview_boundary(source, sec):
        if not source:
            return None
        try:
            duration = inspect_source(source)
            return frame_at(source, min(max(float(sec), 0),
                                        max(0, duration - 0.05)))
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(f"无法显示指定时间的画面：{exc}") from exc

    def choose_strip_frame(source, side, evt: gr.SelectData):
        if not source:
            raise gr.Error("请先上传 MP4 视频")
        try:
            duration = inspect_source(source)
            index = int(evt.index)
            moment = timeline_seconds(duration)[index]
        except (ValueError, TypeError, IndexError) as exc:
            raise gr.Error("无法选择关键帧") from exc
        if side == "起点":
            return gr.update(value=moment), gr.skip()
        return gr.skip(), gr.update(value=moment)

    def mark_this_frame(source, second):
        if not source:
            raise gr.Error("请先上传视频")
        return preview_boundary(source, second)

    def save_frame_markup(editor, source, second, note):
        try:
            marked, metadata, bounds = store_markup(
                editor, source, second, note
            )
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            marked, metadata,
            "✓ 已保存标记帧和 JSON 区域信息。"
            "区域范围：" + ", ".join(f"{n:.2f}" for n in bounds)
            + "。注意：这不是模型自动抠图或真实的 AI 局部重绘。",
        )

    def blur_marked_area(source, editor, first, last, state, capabilities):
        if not source:
            raise gr.Error("请先上传 MP4")
        try:
            rendered, box = local_blur_region(
                source, editor, float(first), float(last)
            )
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        result = attach_video(state, rendered)
        result = append_event(
            result, f"已对固定标记区域执行局部模糊 "
                    f"({float(first):.1f}～{float(last):.1f} 秒)"
        )
        result = choose_mode(result, EDIT)
        status = (
            "✓ 已对视频输出实际模糊效果。"
            "这是固定画面区域，不会追踪移动对象；"
            "已更新预览和当前视频素材。"
        )
        return (
            rendered, rendered, result, _mode_description(result, capabilities),
            *_mode_buttons(EDIT), _submit_button(result, capabilities),
            status, rendered,
        )

    def apply_trim(uploaded, first, last, state, capabilities):
        if not uploaded:
            raise gr.Error("请先上传原视频")
        try:
            cropped = trim_source_video(uploaded, first, last)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        result = attach_video(state, cropped)
        result = append_event(
            result, f"已将原片裁剪为 {first:.2f}～{last:.2f} 秒片段"
        )
        duration = inspect_source(cropped)
        return (
            cropped, cropped, result,
            _mode_description(result, capabilities),
            *_mode_buttons(EDIT), _submit_button(result, capabilities),
            gr.update(value=0, maximum=duration, interactive=True),
            gr.update(value=duration, maximum=duration, interactive=True),
            f"✓ 已裁剪为 {duration:.2f} 秒，预览及后续生成使用裁剪片段。",
            cropped,
        )

    def restore_source(uploaded, state, capabilities):
        if not uploaded:
            raise gr.Error("请先上传原视频")
        result = attach_video(state, uploaded)
        result = append_event(result, "已恢复使用原始视频")
        duration = inspect_source(uploaded)
        return (
            uploaded, uploaded, result, _mode_description(result, capabilities),
            *_mode_buttons(EDIT), _submit_button(result, capabilities),
            gr.update(value=0, maximum=duration, interactive=True),
            gr.update(value=duration, maximum=duration, interactive=True),
            "✓ 已恢复原始上传视频。",
            None,
        )

    def apply_preset(name, state, capabilities):
        selected = choose_edit_preset(name)
        result = choose_mode(state, EDIT)
        return (
            result, selected,
            f"✓ 已填入「{name}」，可在输入框中进一步修改。",
            _mode_description(result, capabilities),
            *_mode_buttons(EDIT),
            _submit_button(result, capabilities),
        )

    def submit_workspace(state, source, prompt, style, shots, capabilities):
        mode = state.get("mode", STORY)
        try:
            action = plan_submission(
                mode, source, prompt, int(shots), capabilities
            )
            if action == VISION:
                path = Path(source)
                if (not path.is_file() or path.suffix.lower() != ".mp4"
                        or path.stat().st_size > 100 * 1024 * 1024):
                    raise ValueError("需要上传不超过 100 MiB 的 MP4")
                with path.open("rb") as stream:
                    plan = send(
                        "POST", "/api/video-understand",
                        files={"file": ("source.mp4", stream, "video/mp4")},
                        data={
                            "instruction": prompt.strip() or "总结视频并规划分镜",
                            "shot_count": str(int(shots)),
                            "theme_style": style,
                        },
                    )
                result = begin_submission(
                    state, VISION, plan,
                    summary="分析上传视频并自动规划分镜",
                )
            elif action == EDIT:
                instruction = video_edit_instruction(prompt)
                path = Path(source)
                if (
                    not path.is_file() or path.suffix.lower() != ".mp4"
                    or path.stat().st_size > 100 * 1024 * 1024
                ):
                    raise ValueError("请选择不超过 100 MiB 的有效 MP4 视频")
                with path.open("rb") as stream:
                    job = send(
                        "POST", "/api/video-edits",
                        files={"file": ("source.mp4", stream, "video/mp4")},
                        data={"prompt": instruction},
                    )
                result = begin_submission(
                    state, EDIT, job,
                    summary="提交视频智能编辑：" + instruction[:75],
                )
            else:
                created = send("POST", "/api/projects", payload={
                    "script": prompt,
                    "theme_style": style,
                    "shot_count": int(shots),
                })
                project = send(
                    "POST",
                    f"/api/projects/{created['project_id']}/generate",
                )
                result = begin_submission(
                    state, STORY, project,
                    summary="创作分镜：" + prompt.strip()[:75],
                )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return (
            result, _mode_description(result, capabilities),
            _submit_button(result, capabilities),
        )

    def refresh_workspace(state, capabilities):
        result = state
        for mode, key, route in (
            (EDIT, "edit_job", "/api/video-edits/"),
            (STORY, "story_project", "/api/projects/"),
        ):
            item = result.get(key) or {}
            if item.get("status") not in {"QUEUED", "GENERATING", "STITCHING"}:
                continue
            identifier = item.get("job_id" if mode == EDIT else "project_id")
            if not identifier:
                continue
            try:
                with httpx.Client(timeout=8) as client:
                    response = client.get(base + route + identifier)
                    response.raise_for_status()
                    updated = response.json()
                result = update_task(result, mode, updated)
            except (httpx.HTTPError, ValueError):
                continue
        if result == state:
            return gr.skip(), gr.skip()
        return result, _submit_button(result, capabilities)

    def generate_analyzed_story(state, capabilities):
        project = state.get("story_project")
        if not state.get("analysis") or not project:
            raise gr.Error("请先运行视频理解获得分镜草稿")
        if has_active_job(state, STORY):
            raise gr.Error("分镜任务正在生成，请等待完成")
        result = send(
            "POST", f"/api/projects/{project['project_id']}/generate"
        )
        state = choose_mode(state, STORY)
        state = update_task(state, STORY, result)
        state = append_event(
            state, "已确认视觉模型生成的分镜，开始生成视频粗剪"
        )
        return (
            state,
            _mode_description(state, capabilities),
            *_mode_buttons(STORY), _submit_button(state, capabilities),
        )

    def regenerate_shot(shot_id: int, state, prompt, duration, motion):
        project = state.get("story_project")
        if not project:
            raise gr.Error("没有可编辑的分镜项目")
        pid = project["project_id"]
        send("PATCH", f"/api/projects/{pid}/shots/{shot_id}", payload={
            "prompt": prompt, "duration": duration, "camera_motion": motion,
        })
        updated = send(
            "POST", f"/api/projects/{pid}/shots/{shot_id}/regenerate"
        )
        result = update_task(state, STORY, updated)
        return append_event(result, f"已提交镜头 {shot_id:02d} 的重新生成")

    # Gradio 6 configures CSS/theme at launch; Gradio 5 does so in Blocks.
    version = int(gr.__version__.split(".")[0])
    options = {
        "title": "Agentic Video Studio · 创作工作台",
        "elem_id": "studio-app",
    }
    if version < 6:
        options["theme"] = gr.themes.Soft(
            primary_hue="violet", neutral_hue="slate",
        )
        options["css"] = STYLE_PATH.read_text(encoding="utf-8")
    with gr.Blocks(**options) as demo:
        workspace = gr.State(initial_workspace())
        selected_source = gr.State(None)
        capability_state = gr.State({
            "connected": False, "video_edit_upload": False,
            "video_understanding": False
        })
        gr.HTML(
            '<div class="studio-app-header">'
            '<div class="studio-app-brand">'
            '<span class="studio-app-logo">✦</span>'
            '<div><strong>AGENTIC VIDEO STUDIO</strong>'
            '<small>Agent-driven creator workspace</small></div></div>'
            '<span class="studio-app-version">MVP · LOCAL WORKSPACE</span>'
            '</div>'
        )
        with gr.Column(elem_classes=["studio-creator-center"]):
            gr.HTML(
                '<div class="studio-creator-title">'
                '<div class="studio-eyebrow">CREATE · EDIT · COMPOSE</div>'
                '<h1>今天想创作什么？</h1>'
                '<p>描述你的想法、上传参考视频，选择合适的创作技能。'
                '在同一个工作台里查看结果并继续调整。</p>'
                '</div>'
            )
            with gr.Group(elem_classes=["studio-composer-card"]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=270):
                        gr.HTML(
                            '<div class="studio-composer-label">'
                            '＋ 添加参考素材</div>'
                        )
                        # The file input stays compact. A read-only preview
                        # deliberately has no Gradio scissors editor: its
                        # buttons were clipped inside this narrow composer.
                        source = gr.File(
                            label="上传参考视频 · MP4",
                            file_types=[".mp4"], file_count="single",
                            type="filepath",
                            elem_classes=["studio-file-upload"],
                        )
                        preview = gr.Video(
                            label="素材预览", interactive=False,
                            height=208,
                            elem_classes=["studio-video-preview"],
                        )
                        gr.HTML(
                            '<div class="studio-composer-hint">'
                            '上传后可预览 · 需要截取片段请展开下方裁剪工具'
                            '</div>'
                        )
                        # Keep errors visible even while the trim panel is closed.
                        trim_status = gr.Markdown(
                            "尚未上传视频 · 拖入 MP4 或点击上传",
                            elem_classes=["studio-trim-status"],
                        )
                    with gr.Column(scale=5, min_width=340):
                        prompt = gr.Textbox(
                            label="你的创作想法",
                            lines=5, max_lines=11,
                            placeholder=(
                                "描述想要创作的视频，或对现有视频做哪些修改…\n"
                                "例如：将上传视频里的白天改成黄昏，"
                                "保持人物、动作和镜头不变。"
                            ),
                            elem_classes=["studio-main-prompt"],
                        )
                gr.HTML('<div class="studio-composer-divider"></div>')
                with gr.Row(equal_height=True, elem_classes=["studio-skill-bar"]):
                    edit_button = gr.Button(
                        "✦ 视频编辑", variant="secondary",
                        elem_classes=["studio-skill-chip"],
                    )
                    story_button = gr.Button(
                        "▦ 创作分镜", variant="primary",
                        elem_classes=["studio-skill-chip"],
                    )
                    vision_button = gr.Button(
                        "◉ 视频理解", variant="secondary",
                        elem_classes=["studio-skill-chip"],
                    )
                    submit_button = gr.Button(
                        "✦ 生成作品 · 正在连接…", variant="primary",
                        interactive=False, elem_classes=["studio-send-button"],
                    )
                capability_notice = gr.Markdown(
                    "正在检测模型服务…",
                    elem_classes=["studio-composer-status"],
                )
            with gr.Accordion(
                "✂ 时间轴裁剪与画面标记 · 查看起止帧 / 局部编辑",
                open=False, elem_classes=["studio-visual-editor"],
            ):
                gr.Markdown(
                    "**时间轴裁剪** · 滑动时间或点击下方关键帧，"
                    "起点和终点的真实视频画面会同步显示。"
                    "应用裁剪会生成新的 MP4，原视频不会被覆盖。"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(min_width=260):
                        trim_start = gr.Slider(
                            0, 30, step=0.1, value=0,
                            label="裁剪起点 · 秒", interactive=False,
                        )
                        start_image = gr.Image(
                            label="起点画面 · START", interactive=False,
                            height=210, elem_classes=["studio-boundary-frame"],
                        )
                    with gr.Column(min_width=260):
                        trim_end = gr.Slider(
                            0, 30, step=0.1, value=10,
                            label="裁剪终点 · 秒", interactive=False,
                        )
                        end_image = gr.Image(
                            label="终点画面 · END", interactive=False,
                            height=210, elem_classes=["studio-boundary-frame"],
                        )
                strip_target = gr.Radio(
                    ["起点", "终点"], value="起点",
                    label="点击关键帧设置", interactive=True,
                )
                frame_strip = gr.Gallery(
                    label="视频关键帧 · 点击某一帧设置时间点",
                    columns=8, rows=1, height=120,
                    object_fit="contain", allow_preview=False,
                    elem_classes=["studio-frame-strip"],
                )
                with gr.Row():
                    trim_button = gr.Button(
                        "✂ 应用裁剪", variant="primary",
                        elem_classes=["studio-secondary-btn"],
                    )
                    restore_button = gr.Button(
                        "↶ 恢复原视频", variant="secondary",
                        elem_classes=["studio-secondary-btn"],
                    )
                gr.Markdown(
                    "**画面标记** · 选择帧时间并载入画板，"
                    "用画笔或橡皮标记希望处理的区域。"
                    "标记可保存为图片/JSON；本地支持对标记区域打模糊，"
                    "其他 AI 改衣服、换背景等仍需要具备区域编辑能力的模型。"
                )
                mark_second = gr.Slider(
                    0, 30, step=0.1, value=0,
                    label="载入标记画面的时间 · 秒", interactive=False,
                )
                mark_load = gr.Button(
                    "↗ 载入该时间的画面并开始标记",
                    elem_classes=["studio-secondary-btn"],
                )
                marker = gr.ImageEditor(
                    label="画面标记画板 · 用画笔涂抹目标区域",
                    height=430, type="pil",
                    sources=[], transforms=[],
                    brush=gr.Brush(
                        colors=["#ff4057"], default_color="#ff4057",
                        color_mode="fixed",
                    ),
                    elem_classes=["studio-mark-canvas"],
                )
                mark_instruction = gr.Textbox(
                    label="区域编辑说明 / Prompt",
                    placeholder="例如：标记区域是背景垃圾桶，希望移除；"
                                "或先尝试本地局部模糊",
                    lines=2,
                )
                with gr.Row():
                    save_mark = gr.Button(
                        "保存标注图片与说明", variant="secondary",
                        elem_classes=["studio-secondary-btn"],
                    )
                    blur_button = gr.Button(
                        "应用本地局部模糊（实际修改视频）",
                        variant="primary",
                        elem_classes=["studio-secondary-btn"],
                    )
                mark_status = gr.Markdown(
                    "尚未载入标记画面。注意：固定区域模糊不会自动跟踪运动物体。",
                    elem_classes=["studio-trim-status"],
                )
                local_export = gr.File(
                    label="下载已裁剪 / 已模糊的本地 MP4",
                    interactive=False,
                )
                with gr.Row():
                    marked_preview = gr.Image(
                        label="保存后的标记画面",
                        interactive=False, height=220,
                    )
                    manifest_file = gr.File(
                        label="标注信息 JSON（可下载）",
                        interactive=False,
                    )
            with gr.Accordion(
                "创作建议与参数 · 可选",
                open=False, elem_classes=["studio-extra-options"],
            ):
                gr.Markdown(
                    "**快速技能模板** · 点击后填入编辑指令，"
                    "不会自动提交推理任务。音频识别、精确区域编辑"
                    "与风格模型路由尚未实现。"
                )
                with gr.Row(elem_classes=["studio-quick-suggestions"]):
                    preset_buttons = []
                    for name, label in PRESET_CARDS:
                        button = gr.Button(
                            label, size="sm", variant="secondary",
                            elem_classes=["studio-suggestion-chip"],
                        )
                        preset_buttons.append((name, button))
                selected_template = gr.Markdown(
                    "还没有使用模板，可自由输入创作内容。"
                )
                with gr.Row():
                    style = gr.Textbox(
                        label="全片视觉风格（创作分镜时生效）",
                        value="cinematic, consistent color palette",
                        scale=3,
                    )
                    shot_count = gr.Slider(
                        1, 5, step=1, value=3,
                        label="分镜数量（创作分镜时生效）",
                        scale=2,
                    )
                detect_button = gr.Button(
                    "↻ 重新检测后端能力", size="sm",
                    variant="secondary", elem_classes=["studio-secondary-btn"],
                )
            gr.HTML(
                '<div class="studio-under-composer">'
                '✦ 已实现：关键帧视频理解（须多模态 API）、视频条件编辑（须 V2V 模型）、'
                '剧本拆镜与粗剪 · 未实现：视频反解、区域蒙版、'
                '跨任务自动 Agent 编排</div>'
            )

        with gr.Column(elem_classes=["studio-workbench"]):
            gr.HTML(
                '<div class="studio-workbench-heading">'
                '<div><div class="studio-eyebrow">YOUR PROJECT</div>'
                '<h2>当前创作项目</h2>'
                '<p>所有任务都在下方展开，无需切换页面。'
                '创作记录仅保存在当前浏览器会话中。</p></div></div>'
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=315):
                    with gr.Group(elem_classes=["studio-panel"]):
                        gr.HTML(
                            '<div class="studio-card-title">✦ 创作动态</div>'
                        )
                        @gr.render(inputs=workspace)
                        def timeline(state):
                            gr.HTML(_timeline_html(state.get("timeline", []),
                                                   state.get("mode", STORY)))
                        refresh_button = gr.Button(
                            "↻ 刷新任务状态", size="sm",
                            elem_classes=["studio-secondary-btn"],
                        )
                with gr.Column(scale=5, min_width=430):
                    with gr.Group(elem_classes=["studio-panel"]):
                        gr.HTML(
                            '<div class="studio-card-title">▶ 作品预览与结果</div>'
                        )
                        @gr.render(inputs=[workspace, selected_source])
                        def result_panel(state, uploaded):
                            mode = state.get("active", STORY)
                            if mode == VISION:
                                analysis = state.get("analysis")
                                gr.HTML(_analysis_html(analysis))
                                if analysis:
                                    project = analysis.get("project") or {}
                                    gr.Markdown(
                                        "**结构化分镜草稿** · "
                                        "分镜来自视频关键帧而非音频。"
                                        "可在下方修改镜头提示词。"
                                    )
                                    for shot in project.get("shots", []):
                                        gr.Markdown(
                                            f"**镜头 {shot['id']:02d}**："
                                            + str(shot.get("prompt", ""))[:800]
                                        )
                                    if project.get("status") not in {
                                        "GENERATING", "STITCHING", "READY"
                                    }:
                                        generate_plan_button = gr.Button(
                                            "▶ 确认分镜并生成粗剪",
                                            variant="primary",
                                            elem_classes=["studio-send-button"],
                                        )
                                        generate_plan_button.click(
                                            fn=generate_analyzed_story,
                                            inputs=[workspace, capability_state],
                                            outputs=[
                                                workspace, capability_notice,
                                                edit_button, story_button,
                                                vision_button, submit_button,
                                            ],
                                        )
                            elif mode == EDIT:
                                job = state.get("edit_job")
                                gr.HTML(_status_chip(job))
                                with gr.Row():
                                    with gr.Column(min_width=200):
                                        gr.HTML(
                                            '<div class="studio-media-label">'
                                            '原始视频 · BEFORE</div>'
                                        )
                                        if job:
                                            gr.Video(
                                                value=base + "/api/video-edits/"
                                                + job["job_id"] + "/source",
                                                label="原视频", interactive=False,
                                                height=255,
                                            )
                                        elif uploaded:
                                            gr.Video(
                                                value=uploaded, label="原视频",
                                                interactive=False, height=255,
                                            )
                                        else:
                                            gr.HTML(_empty_preview(
                                                "等待添加视频",
                                                "在上方上传 MP4 原片",
                                            ))
                                    with gr.Column(min_width=200):
                                        gr.HTML(
                                            '<div class="studio-media-label">'
                                            '编辑结果 · AFTER</div>'
                                        )
                                        if job and job.get("status") == "READY":
                                            gr.Video(
                                                value=base + "/api/video-edits/"
                                                + job["job_id"] + "/result",
                                                label="修改后视频",
                                                interactive=False, height=255,
                                            )
                                            url = (
                                                base + "/api/video-edits/"
                                                + job["job_id"] + "/result?download=1"
                                            )
                                            gr.Markdown(f"[↓ 下载编辑结果 MP4]({url})")
                                        else:
                                            gr.HTML(_empty_preview(
                                                "等待编辑结果",
                                                "模型执行完成后在这里对比并下载",
                                                "✦",
                                            ))
                            else:
                                project = state.get("story_project")
                                gr.HTML(_status_chip(project))
                                if project and project.get("status") == "READY":
                                    url = (
                                        base + "/api/projects/"
                                        + project["project_id"] + "/video"
                                    )
                                    gr.Video(
                                        value=url, label="粗剪预览",
                                        interactive=False, height=285,
                                    )
                                    gr.Markdown(f"[↓ 下载粗剪 MP4]({url}?download=1)")
                                else:
                                    gr.HTML(_empty_preview(
                                        "粗剪将在这里出现",
                                        "提交剧本后自动解析分镜、生成镜头并合成",
                                        "▦",
                                    ))
                    @gr.render(inputs=workspace)
                    def storyboard_assets(state):
                        project = state.get("story_project")
                        if state.get("active") not in {STORY, VISION} or not project:
                            return
                        gr.HTML(
                            '<div class="studio-assets-title">'
                            '<h3>▦ 分镜资产</h3>'
                            '<p>逐镜头调整并重新生成，不影响其他已就绪片段。</p>'
                            '</div>'
                        )
                        for shot in project["shots"]:
                            sid = shot["id"]
                            with gr.Group(elem_classes=["studio-shot-card"]):
                                label = STATUS_LABELS.get(
                                    shot["status"], (shot["status"], "")
                                )[0]
                                gr.HTML(
                                    '<div class="studio-shot-header">'
                                    f'<span class="studio-shot-badge">SHOT {sid:02d}</span>'
                                    f'<b>{html.escape(label)}</b>'
                                    f'<span>版本 {shot["revision"]} / '
                                    f'已渲染 {shot["rendered_revision"]}</span>'
                                    '</div>'
                                )
                                with gr.Row():
                                    with gr.Column(scale=3, min_width=245):
                                        shot_prompt = gr.Textbox(
                                            value=shot["prompt"], label="镜头描述",
                                            lines=3, key=f"shot-prompt-{sid}",
                                        )
                                        with gr.Row():
                                            duration = gr.Slider(
                                                1, 10, step=0.5,
                                                value=shot["duration"],
                                                label="目标时长 / 秒",
                                                key=f"shot-duration-{sid}",
                                            )
                                            motions = [
                                                "static", "pan", "tilt", "tracking",
                                                "dolly", "handheld",
                                            ]
                                            motion = gr.Dropdown(
                                                motions,
                                                value=shot["camera_motion"]
                                                if shot["camera_motion"] in motions
                                                else "static",
                                                label="运镜",
                                                key=f"shot-motion-{sid}",
                                            )
                                        rerun = gr.Button(
                                            "↻ 修改并重投当前镜头",
                                            key=f"rerun-{sid}",
                                            variant="secondary",
                                            elem_classes=["studio-secondary-btn"],
                                        )
                                        rerun.click(
                                            fn=partial(regenerate_shot, sid),
                                            inputs=[
                                                workspace, shot_prompt,
                                                duration, motion,
                                            ],
                                            outputs=workspace,
                                        )
                                    with gr.Column(scale=2, min_width=225):
                                        if (
                                            shot["status"] == "READY"
                                            and shot["rendered_revision"] ==
                                            shot["revision"] and shot["video_path"]
                                        ):
                                            gr.Video(
                                                value=(
                                                    base + "/api/projects/"
                                                    + project["project_id"]
                                                    + f"/shots/{sid}/video"
                                                ),
                                                label="镜头预览",
                                                interactive=False, height=215,
                                            )
                                        else:
                                            gr.HTML(_empty_preview(
                                                "等待镜头生成",
                                                str(shot.get("error")
                                                    or "完成后可以预览此镜头"),
                                                "▶",
                                            ))

        gr.HTML(
            '<footer class="studio-editor-footer">'
            'AGENTIC VIDEO STUDIO · 独立创作工作台，并非即梦 AI 官方服务。'
            '当前会话统一展示任务；后端视频编辑与分镜项目仍分别持久化。'
            'Mock 不支持真实 AI 编辑。</footer>'
        )

        edit_button.click(
            fn=partial(select_mode, mode=EDIT),
            inputs=[workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
            ],
        )
        vision_button.click(
            fn=partial(select_mode, mode=VISION),
            inputs=[workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
            ],
        )
        story_button.click(
            fn=partial(select_mode, mode=STORY),
            inputs=[workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
            ],
        )
        source.change(
            fn=upload_changed,
            inputs=[source, workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
                selected_source, preview, trim_start, trim_end, trim_status,
                local_export,
            ],
        )
        # A gallery frame explicitly sets one of the two time boundaries.
        frame_strip.select(
            fn=choose_strip_frame,
            inputs=[selected_source, strip_target],
            outputs=[trim_start, trim_end],
        )
        trim_start.release(
            fn=preview_boundary,
            inputs=[selected_source, trim_start],
            outputs=start_image,
        )
        trim_end.release(
            fn=preview_boundary,
            inputs=[selected_source, trim_end],
            outputs=end_image,
        )
        selected_source.change(
            fn=setup_visual_timeline,
            inputs=[selected_source, trim_start, trim_end],
            outputs=[
                frame_strip, start_image, end_image,
                mark_second, marker, mark_status,
            ],
        )
        mark_load.click(
            fn=mark_this_frame,
            inputs=[selected_source, mark_second],
            outputs=marker,
        )
        save_mark.click(
            fn=save_frame_markup,
            inputs=[marker, selected_source, mark_second, mark_instruction],
            outputs=[marked_preview, manifest_file, mark_status],
        )
        blur_button.click(
            fn=blur_marked_area,
            inputs=[
                selected_source, marker, trim_start, trim_end,
                workspace, capability_state,
            ],
            outputs=[
                selected_source, preview, workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
                mark_status, local_export,
            ],
        )
        trim_button.click(
            fn=apply_trim,
            inputs=[
                selected_source, trim_start, trim_end, workspace, capability_state,
            ],
            outputs=[
                selected_source, preview, workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
                trim_start, trim_end, trim_status, local_export,
            ],
        )
        restore_button.click(
            fn=restore_source,
            inputs=[source, workspace, capability_state],
            outputs=[
                selected_source, preview, workspace, capability_notice,
                edit_button, story_button, vision_button, submit_button,
                trim_start, trim_end, trim_status, local_export,
            ],
        )
        for name, button in preset_buttons:
            button.click(
                fn=partial(apply_preset, name),
                inputs=[workspace, capability_state],
                outputs=[
                    workspace, prompt, selected_template, capability_notice,
                    edit_button, story_button, vision_button, submit_button,
                ],
            )
        submit_button.click(
            fn=submit_workspace,
            inputs=[
                workspace, selected_source, prompt, style, shot_count, capability_state,
            ],
            outputs=[workspace, capability_notice, submit_button],
        )
        demo.load(
            fn=detect_capabilities,
            inputs=workspace,
            outputs=[capability_state, capability_notice, submit_button],
        )
        detect_button.click(
            fn=detect_capabilities,
            inputs=workspace,
            outputs=[capability_state, capability_notice, submit_button],
        )
        refresh_button.click(
            fn=refresh_workspace,
            inputs=[workspace, capability_state],
            outputs=[workspace, submit_button],
        )
        timer = gr.Timer(3)
        timer.tick(
            fn=refresh_workspace,
            inputs=[workspace, capability_state],
            outputs=[workspace, submit_button],
        )
    return demo


def main():
    import gradio as gr
    options = {"server_name": "127.0.0.1", "server_port": 7860}
    if int(gr.__version__.split(".")[0]) >= 6:
        options["theme"] = gr.themes.Soft(
            primary_hue="violet", neutral_hue="slate",
        )
        options["css"] = STYLE_PATH.read_text(encoding="utf-8")
    build_ui().launch(**options)


if __name__ == "__main__":
    main()
