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

from .workspace import (
    EDIT, STORY, append_event, attach_video, begin_submission, choose_mode,
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


def _mode_description(state: dict, capabilities: dict) -> str:
    mode = state.get("mode", STORY)
    connected = capabilities.get("connected", False)
    if not connected:
        return "尚未连接 API，请先启动后端。"
    if mode == EDIT:
        notice, _ = video_edit_availability(capabilities)
        return notice
    if state.get("source_name"):
        return "创作分镜当前不使用视频参考，请先移除上传的 MP4 或切换视频编辑。"
    return "剧本 → 分镜 → 镜头生成 → 粗剪合成。Mock 模式也能体验完整流程。"


def _submit_button(state: dict, capabilities: dict):
    import gradio as gr
    mode = state.get("mode", STORY)
    connected = bool(capabilities.get("connected"))
    if not connected:
        return gr.update(interactive=False, value="API 未连接")
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
                    "model_v2v_support_verified":
                        bool(data.get("model_v2v_support_verified")),
                }
        except (httpx.HTTPError, ValueError):
            capabilities = {
                "connected": False, "video_edit_upload": False
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

    def upload_changed(source, state, capabilities):
        result = attach_video(state, source)
        return (
            result, _mode_description(result, capabilities),
            *_mode_buttons(result["mode"]),
            _submit_button(result, capabilities),
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
            if action == EDIT:
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

    with gr.Blocks(
        title="Agentic Video Studio · 创作工作台",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css=STYLE_PATH.read_text(encoding="utf-8"),
        elem_id="studio-app",
    ) as demo:
        workspace = gr.State(initial_workspace())
        capability_state = gr.State({
            "connected": False, "video_edit_upload": False
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
                    with gr.Column(scale=1, min_width=165):
                        gr.HTML(
                            '<div class="studio-composer-label">'
                            '＋ 添加参考素材</div>'
                        )
                        source = gr.File(
                            label="上传视频 · MP4",
                            file_types=[".mp4"], type="filepath",
                            elem_classes=["studio-compact-upload"],
                        )
                        gr.HTML(
                            '<div class="studio-composer-hint">'
                            '仅视频编辑使用原片 · ≤30 秒 / 100 MiB</div>'
                        )
                    with gr.Column(scale=4, min_width=280):
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
                    submit_button = gr.Button(
                        "正在连接 API…", variant="primary",
                        interactive=False, elem_classes=["studio-send-button"],
                    )
                capability_notice = gr.Markdown(
                    "正在检测模型服务…",
                    elem_classes=["studio-composer-status"],
                )
            with gr.Accordion(
                "创作建议与参数 · 可选",
                open=False, elem_classes=["studio-extra-options"],
            ):
                gr.Markdown(
                    "**快速技能模板** · 点击后填入编辑指令，"
                    "不会自动提交推理任务。视频反解、精确区域编辑"
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
                '✦ 已实现：视频条件编辑（须 V2V 模型）、'
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
                        @gr.render(inputs=[workspace, source])
                        def result_panel(state, uploaded):
                            mode = state.get("active", STORY)
                            if mode == EDIT:
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
                        if state.get("active") != STORY or not project:
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
                edit_button, story_button, submit_button,
            ],
        )
        story_button.click(
            fn=partial(select_mode, mode=STORY),
            inputs=[workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, submit_button,
            ],
        )
        source.change(
            fn=upload_changed,
            inputs=[source, workspace, capability_state],
            outputs=[
                workspace, capability_notice,
                edit_button, story_button, submit_button,
            ],
        )
        for name, button in preset_buttons:
            button.click(
                fn=partial(apply_preset, name),
                inputs=[workspace, capability_state],
                outputs=[
                    workspace, prompt, selected_template, capability_notice,
                    edit_button, story_button, submit_button,
                ],
            )
        submit_button.click(
            fn=submit_workspace,
            inputs=[
                workspace, source, prompt, style, shot_count, capability_state,
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
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
