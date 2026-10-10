"""Creator-first Gradio workbench: smart edit and storyboard rough-cut.

The UI keeps Gradio controls for real interactions and uses static HTML/CSS
only for layout and presentation. Model V2V support is never assumed.
"""
from __future__ import annotations

import html
from functools import partial
import os
from pathlib import Path

import httpx

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

# Each suggestion is a real button, not a dropdown option or an inert HTML card.
PRESET_CARDS = (
    ("白天变黄昏 · 推荐首测", "🌇  光影氛围", "白天 → 黄昏"),
    ("替换桌上马克杯", "☕  替换物体", "马克杯 → 玻璃杯"),
    ("移除背景垃圾桶", "✦  清理画面", "移除指定物体"),
    ("缓慢推近镜头 · 探索", "◈  镜头运动", "固定机位 → 缓慢推进"),
)

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "studio.css"
STATUS_LABELS = {
    "QUEUED": ("排队中", "running"),
    "GENERATING": ("生成中", "running"),
    "READY": ("已完成", "ready"),
    "FAILED": ("生成失败", "failed"),
    "DRAFT": ("待生成", ""),
    "STITCHING": ("正在合成", "running"),
    "PENDING": ("等待生成", ""),
}


def choose_edit_preset(name: str) -> str:
    return EDIT_PRESETS.get(name, "")


def selected_preset_cards(name: str) -> tuple[str, str, tuple[bool, ...]]:
    """Select one template without submitting a render or changing the media."""
    if name not in EDIT_PRESETS:
        raise ValueError("Unknown edit template")
    return (
        EDIT_PRESETS[name],
        f"✓ 已选择「{name}」。下方指令可继续自由编辑。",
        tuple(item[0] == name for item in PRESET_CARDS),
    )


def video_edit_instruction(prompt: str) -> str:
    """Validate instruction without silently rewriting preservation constraints."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("请先输入编辑指令")
    if len(prompt) > 3000:
        raise ValueError("编辑指令过长（最多 3000 字符）")
    return prompt


def video_edit_availability(info: dict) -> tuple[str, bool]:
    """Distinguish a configured API from verified V2V model capabilities."""
    if not info.get("video_edit_upload"):
        return (
            "**当前是 Mock 模式 · 智能编辑暂不可用。** "
            "可以上传素材、调整指令并熟悉工作台；真正修改原视频需要部署 "
            "V2V 模型，设置 AVG_BACKEND=omni 与 AVG_OMNI_URL，然后重启 API。"
            "需要立即体验现有能力，请切换到「剧本成片」。",
            False,
        )
    return (
        "**vLLM-Omni 推理接口已配置。** 可以提交编辑任务，"
        "但不代表实际模型支持 V2V 的 input_reference；"
        "请先确认模型能力，原片人物、动作和时序能否保持取决于模型。",
        True,
    )


def _panel_head(step: str, title: str, detail: str = "") -> str:
    return (
        '<div class="studio-pane-title">'
        f'<span class="studio-pane-index">{html.escape(step)}</span>'
        '<div>'
        f'<h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(detail)}</p>'
        '</div></div>'
    )


def _empty_preview(title: str, detail: str, mark: str = "▶") -> str:
    return (
        '<div class="studio-result-shell">'
        f'<div class="empty-mark">{html.escape(mark)}</div>'
        f'<strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(detail)}</span>'
        '</div>'
    )


def _job_status(state: dict) -> str:
    status = str(state.get("status", "DRAFT"))
    label, flavor = STATUS_LABELS.get(status, (status, ""))
    identifier = html.escape(str(state.get("job_id") or state.get("project_id") or "")[:8])
    error = html.escape(str(state.get("error") or ""))
    return (
        '<div class="studio-result-meta">'
        f'<span class="studio-status-chip {flavor}">{html.escape(label)}</span>'
        f'<small>任务 {identifier}</small>'
        '</div>'
        + (f'<p class="studio-hint">错误：{error}</p>' if error else "")
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
            raise gr.Error(f"请求失败（{exc.response.status_code}）：{detail}") from exc
        except httpx.RequestError as exc:
            raise gr.Error(f"无法访问视频 API（{base}）：{exc}") from exc

    def capabilities():
        try:
            with httpx.Client(timeout=8) as client:
                response = client.get(base + "/api/capabilities")
                response.raise_for_status()
                info = response.json()
            notice, enabled = video_edit_availability(info)
            return notice, gr.update(
                interactive=enabled,
                value="开始智能编辑 →" if enabled else "智能编辑暂不可用 · 请先配置模型",
            )
        except (httpx.HTTPError, ValueError):
            return (
                "**API 未连接。** 启动 agentic-video-api 并确认 "
                "http://127.0.0.1:8000/healthz 可访问，然后重新检测。",
                gr.update(
                    interactive=False, value="智能编辑暂不可用 · API 未连接"
                ),
            )

    def start_edit(source: str | None, prompt: str, existing: dict | None):
        if existing and existing.get("status") in {"QUEUED", "GENERATING"}:
            raise gr.Error("当前编辑任务未完成，请等待当前任务结束")
        if not source:
            raise gr.Error("请先上传一段 MP4 原视频")
        try:
            valid_prompt = video_edit_instruction(prompt)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        path = Path(source)
        if (
            path.suffix.lower() != ".mp4" or not path.is_file()
            or path.stat().st_size > 100 * 1024 * 1024
        ):
            raise gr.Error("仅支持小于 100 MiB 的本地 MP4 文件")
        with path.open("rb") as stream:
            return send(
                "POST", "/api/video-edits",
                files={"file": ("source.mp4", stream, "video/mp4")},
                data={"prompt": valid_prompt},
            )

    def update_edit(job: dict | None):
        if not job or job.get("status") not in {"QUEUED", "GENERATING"}:
            return gr.skip()
        try:
            with httpx.Client(timeout=8) as client:
                response = client.get(base + f"/api/video-edits/{job['job_id']}")
                response.raise_for_status()
                latest = response.json()
            return latest if latest != job else gr.skip()
        except (httpx.HTTPError, ValueError):
            return gr.skip()

    def create_story(script: str, style: str, count: int):
        if not script.strip():
            raise gr.Error("请输入剧本或口播文案")
        return send("POST", "/api/projects", payload={
            "script": script, "theme_style": style, "shot_count": int(count),
        })

    def start_story(project: dict | None):
        if not project:
            raise gr.Error("请先解析分镜")
        return send("POST", f"/api/projects/{project['project_id']}/generate")

    def update_story(project: dict | None):
        if not project or project.get("status") not in {
            "DRAFT", "GENERATING", "STITCHING"
        }:
            return gr.skip()
        try:
            with httpx.Client(timeout=8) as client:
                response = client.get(
                    base + f"/api/projects/{project['project_id']}"
                )
                response.raise_for_status()
                latest = response.json()
            return latest if latest != project else gr.skip()
        except (httpx.HTTPError, ValueError):
            return gr.skip()

    def retry_shot(shot_id: int):
        def handler(project, prompt, duration, motion):
            if not project:
                raise gr.Error("请先创建分镜工程")
            pid = project["project_id"]
            send("PATCH", f"/api/projects/{pid}/shots/{shot_id}", payload={
                "prompt": prompt,
                "duration": duration,
                "camera_motion": motion,
            })
            return send(
                "POST", f"/api/projects/{pid}/shots/{shot_id}/regenerate"
            )
        return handler

    with gr.Blocks(
        title="Agentic Video Studio",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css=STYLE_PATH.read_text(encoding="utf-8"),
        elem_id="studio-app",
    ) as demo:
        gr.HTML(
            '<header class="studio-hero">'
            '<div class="studio-brand">'
            '<span class="studio-brand-mark">✦</span>'
            '<span class="studio-brand-name">AGENTIC VIDEO STUDIO</span>'
            '<span class="studio-beta">WORKSPACE · BETA</span>'
            '</div>'
            '<h1>让每一个镜头，都恰如你所想。</h1>'
            '<p>视频编辑和剧本成片，一站式完成。'
            '从素材、创意描述到可预览的作品，把创作流程留在一个工作台。</p>'
            '<div class="studio-layout-label">VIDEO EDITING  /  STORYBOARD  /  ROUGH CUT</div>'
            '</header>'
        )
        with gr.Tabs(elem_classes=["studio-workflow-tabs"]):
            with gr.Tab("✦  视频智能编辑", id="edit"):
                gr.HTML(
                    '<div class="studio-section-head">'
                    '<div class="studio-eyebrow">VIDEO TO VIDEO · SMART EDIT</div>'
                    '<h2>编辑已有视频</h2>'
                    '<p>保留原片作为输入，告诉模型希望调整的部分。'
                    '建议使用 5～10 秒短片初次体验。</p></div>'
                    '<div class="studio-step-track">'
                    '<span class="step"><b>01</b>上传素材</span><span class="step-arrow">→</span>'
                    '<span class="step"><b>02</b>输入指令</span><span class="step-arrow">→</span>'
                    '<span class="step"><b>03</b>生成结果</span><span class="step-arrow">→</span>'
                    '<span class="step"><b>04</b>对比导出</span></div>'
                )
                edit_state = gr.State({})
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, min_width=390):
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "01", "上传你的视频", "支持 MP4 · 最长 30 秒 · 不超过 100 MiB"
                            ))
                            original = gr.Video(
                                label="原始素材",
                                sources=["upload"], format="mp4",
                                interactive=True, height=242,
                                elem_classes=["studio-video-input"],
                            )
                            gr.HTML(
                                '<p class="studio-hint">提示：尽量选择主体清晰、'
                                '镜头运动简单的片段，便于对照编辑效果。</p>'
                            )
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "02", "描述你想要的修改",
                                "点击任一灵感卡片，直接填充下方的编辑指令"
                            ))
                            gr.HTML('<div class="studio-card-title">✨ 试试这些编辑场景</div>')
                            preset_buttons = []
                            for i in range(0, len(PRESET_CARDS), 2):
                                with gr.Row(elem_classes=["studio-preset-grid"]):
                                    for key, heading, tagline in PRESET_CARDS[i:i+2]:
                                        with gr.Column(min_width=150):
                                            button = gr.Button(
                                                f"{heading}\n{tagline}",
                                                variant="primary" if key == PRESET_CARDS[0][0]
                                                else "secondary",
                                                size="md",
                                                elem_classes=["studio-preset-card"],
                                            )
                                            preset_buttons.append((key, button))
                            selected = gr.Markdown(
                                "✓ 已选择「白天变黄昏 · 推荐首测」。下方指令可继续自由编辑。",
                                elem_classes=["studio-hint"],
                            )
                            instruction = gr.Textbox(
                                label="编辑指令 / Prompt",
                                value=EDIT_PRESETS["白天变黄昏 · 推荐首测"],
                                lines=5,
                                max_lines=11,
                                placeholder="描述要改变什么，以及必须保持不变的内容…",
                                elem_classes=["studio-textarea"],
                            )
                            def select_template(name: str):
                                prompt, caption, highlighted = selected_preset_cards(name)
                                return (
                                    prompt,
                                    caption,
                                    *[
                                        gr.update(
                                            variant="primary" if active else "secondary"
                                        )
                                        for active in highlighted
                                    ],
                                )

                            for key, button in preset_buttons:
                                button.click(
                                    fn=partial(select_template, key),
                                    inputs=[],
                                    outputs=[instruction, selected] + [
                                        item_button for _, item_button in preset_buttons
                                    ],
                                    show_progress="hidden",
                                )
                            gr.HTML(
                                '<p class="studio-hint">写清楚要改变的对象，'
                                '并说明人物、动作、背景或时长中哪些必须保持不变。'
                                '提示词不保证模型一定能严格遵守。</p>'
                            )
                            with gr.Accordion(
                                "高级编辑 · 区域 / 时间定位（尚未支持）",
                                open=False,
                                elem_classes=["studio-accordion"],
                            ):
                                gr.Markdown(
                                    "区域框选、Mask 与精确时间段编辑需要"
                                    "模型和后端的专门支持。目前暂不开放无效操作。"
                                )
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "03", "生成编辑结果", "先确认推理服务已就绪"
                            ))
                            capability_notice = gr.Markdown(
                                "正在检测后端能力…",
                                elem_classes=["studio-status-panel"],
                            )
                            detect_button = gr.Button(
                                "↻ 重新检测服务",
                                variant="secondary",
                                elem_classes=["studio-secondary-btn"],
                            )
                            submit_edit = gr.Button(
                                "检测服务中…",
                                variant="primary", size="lg",
                                interactive=False,
                                elem_classes=["studio-primary-btn"],
                            )
                            gr.HTML(
                                '<p class="studio-hint">模型需支持 V2V / '
                                'input_reference。本项目不提供积分或费用估算。</p>'
                            )
                    with gr.Column(scale=7, min_width=390):
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "04", "预览与对比", "对比光照、人物、动作和时序是否符合要求"
                            ))
                            @gr.render(inputs=[original, edit_state])
                            def edit_preview(source, job):
                                if job:
                                    gr.HTML(_job_status(job))
                                else:
                                    gr.HTML(
                                        '<div class="studio-result-meta">'
                                        '<span class="studio-status-chip">等待提交</span>'
                                        '<small>选择原视频与编辑指令后开始</small></div>'
                                    )
                                with gr.Row(equal_height=True):
                                    with gr.Column(min_width=205):
                                        gr.HTML(
                                            '<div class="studio-card-title">'
                                            '原始视频 · BEFORE</div>'
                                        )
                                        if job:
                                            gr.Video(
                                                value=base + f"/api/video-edits/"
                                                f"{job['job_id']}/source",
                                                label="原片", interactive=False,
                                                height=243,
                                                elem_classes=["studio-compare-video"],
                                            )
                                        elif source:
                                            gr.Video(
                                                value=source, label="原片",
                                                interactive=False, height=243,
                                                elem_classes=["studio-compare-video"],
                                            )
                                        else:
                                            gr.HTML(_empty_preview(
                                                "原视频将在这里显示",
                                                "上传 MP4 后即可开始检查素材", "▣"
                                            ))
                                    with gr.Column(min_width=205):
                                        gr.HTML(
                                            '<div class="studio-card-title">'
                                            '编辑结果 · AFTER</div>'
                                        )
                                        if job and job.get("status") == "READY":
                                            gr.Video(
                                                value=base + f"/api/video-edits/"
                                                f"{job['job_id']}/result",
                                                label="编辑结果", interactive=False,
                                                height=243,
                                                elem_classes=["studio-compare-video"],
                                            )
                                        else:
                                            status = (job or {}).get("status")
                                            hint = (
                                                "请检查下方的错误信息并重试"
                                                if status == "FAILED"
                                                else "生成完成后，编辑后的视频将出现在这里"
                                            )
                                            gr.HTML(_empty_preview(
                                                "编辑结果待生成", hint, "✦"
                                            ))
                                if job and job.get("status") == "READY":
                                    url = (
                                        f"{base}/api/video-edits/"
                                        f"{job['job_id']}/result?download=1"
                                    )
                                    gr.Markdown(
                                        f"**[↓ 下载编辑后的 MP4]({url})**",
                                        elem_classes=["studio-download-link"],
                                    )
                                gr.HTML(
                                    '<p class="studio-hint">'
                                    '检查建议：分别观察开头、中间和结尾画面，'
                                    '确认主体身份、位置、动作、画幅与时长。</p>'
                                )

            with gr.Tab("▦  剧本成片", id="storyboard"):
                gr.HTML(
                    '<div class="studio-section-head">'
                    '<div class="studio-eyebrow">SCRIPT TO ROUGH CUT</div>'
                    '<h2>从剧本到完整粗剪</h2>'
                    '<p>自动拆解分镜，逐镜头生成视频；只改一处，'
                    '只重投对应镜头。</p></div>'
                )
                story_state = gr.State({})
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=350):
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "01", "编写剧本", "设置内容、全片风格与分镜数量"
                            ))
                            script = gr.Textbox(
                                label="剧本 / 口播文案",
                                lines=6,
                                placeholder="例如：清晨，咖啡师打开咖啡馆。热气从杯中升起，顾客接过咖啡微笑。",
                                elem_classes=["studio-textarea"],
                            )
                            style = gr.Textbox(
                                label="统一视觉风格",
                                value="cinematic, consistent color palette",
                            )
                            shot_count = gr.Slider(
                                1, 5, value=3, step=1, label="分镜数量"
                            )
                            parse_button = gr.Button(
                                "解析分镜 →", variant="primary",
                                elem_classes=["studio-primary-btn"],
                            )
                            generate_button = gr.Button(
                                "开始生成并合成",
                                elem_classes=["studio-secondary-btn"],
                            )
                    with gr.Column(scale=7, min_width=360):
                        with gr.Group(elem_classes=["studio-panel"]):
                            gr.HTML(_panel_head(
                                "02", "粗剪预览", "生成完成后可播放并导出 MP4"
                            ))
                            refresh_button = gr.Button(
                                "↻ 刷新项目进度",
                                size="sm",
                                elem_classes=["studio-secondary-btn"],
                            )
                            @gr.render(inputs=story_state)
                            def final_preview(project):
                                if not project:
                                    gr.HTML(_empty_preview(
                                        "这里将出现你的粗剪视频",
                                        "先输入剧本，解析分镜并开始生成", "▤"
                                    ))
                                    return
                                gr.HTML(_job_status(project))
                                if project["status"] == "READY":
                                    url = (
                                        f"{base}/api/projects/"
                                        f"{project['project_id']}/video"
                                    )
                                    gr.Video(
                                        value=url,
                                        label="当前粗剪", interactive=False,
                                        elem_classes=["studio-compare-video"],
                                    )
                                    gr.Markdown(
                                        f"**[↓ 下载粗剪视频]({url}?download=1)**",
                                        elem_classes=["studio-download-link"],
                                    )
                                else:
                                    gr.HTML(_empty_preview(
                                        "等待镜头合成",
                                        "已生成的镜头会保留，修改后不展示过时的粗剪",
                                        "✦",
                                    ))
                gr.HTML(
                    '<div class="studio-section-head" style="margin-top:20px">'
                    '<div class="studio-eyebrow">SHOT WORKSPACE</div>'
                    '<h2>分镜工作台</h2>'
                    '<p>在单张卡片中修改 Prompt、时长或运镜，'
                    '并且只重投这一镜头。</p></div>'
                )
                @gr.render(inputs=story_state)
                def shot_cards(project):
                    if not project:
                        gr.HTML(_empty_preview(
                            "还没有分镜",
                            "点击「解析分镜」后，将出现可逐个编辑的镜头卡片",
                            "▦",
                        ))
                        return
                    for shot in project["shots"]:
                        with gr.Group(elem_classes=["studio-shot-card"]):
                            sid = shot["id"]
                            name, _ = STATUS_LABELS.get(
                                shot["status"], (shot["status"], "")
                            )
                            gr.HTML(
                                '<div class="studio-shot-header">'
                                f'<span class="studio-shot-badge">SHOT {sid:02d}</span>'
                                f'<b>{html.escape(name)}</b>'
                                f'<span>版本 {shot["revision"]} '
                                f'/ 已渲染 {shot["rendered_revision"]}</span></div>'
                            )
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=6, min_width=260):
                                    shot_prompt = gr.Textbox(
                                        value=shot["prompt"],
                                        label="镜头描述",
                                        lines=3,
                                        key=f"shot-prompt-{sid}",
                                        elem_classes=["studio-textarea"],
                                    )
                                    with gr.Row():
                                        duration = gr.Slider(
                                            1, 10, value=shot["duration"],
                                            step=0.5, label="时长 · 秒",
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
                                            label="运镜", key=f"shot-motion-{sid}",
                                        )
                                    rerun = gr.Button(
                                        "↻ 保存修改并重新生成此镜头",
                                        key=f"shot-rerun-{sid}",
                                        elem_classes=["studio-secondary-btn"],
                                    )
                                    rerun.click(
                                        retry_shot(sid),
                                        [story_state, shot_prompt, duration, motion],
                                        story_state,
                                    )
                                with gr.Column(scale=5, min_width=230):
                                    if (
                                        shot["status"] == "READY"
                                        and shot["rendered_revision"]
                                        == shot["revision"]
                                        and shot["video_path"]
                                    ):
                                        gr.Video(
                                            value=base + f"/api/projects/"
                                            f"{project['project_id']}/shots/"
                                            f"{sid}/video",
                                            label="镜头预览",
                                            interactive=False, height=235,
                                            elem_classes=["studio-compare-video"],
                                        )
                                    else:
                                        gr.HTML(_empty_preview(
                                            "镜头待生成",
                                            str(shot.get("error") or "点击生成后在此预览"),
                                            "▶",
                                        ))
                parse_button.click(
                    create_story, [script, style, shot_count], story_state,
                )
                generate_button.click(start_story, story_state, story_state)
                refresh_button.click(update_story, story_state, story_state)
        gr.HTML(
            '<footer class="studio-editor-footer"><p>'
            'AGENTIC VIDEO STUDIO · 独立创作工具，非即梦 AI 官方服务。'
            '智能视频编辑依赖 V2V 模型，区域 Mask 和精确时间段编辑尚未接入。'
            'Mock 模式仅供分镜技术流程验证。</p></footer>'
        )
        submit_edit.click(
            start_edit, [original, instruction, edit_state], edit_state,
        )
        demo.load(
            capabilities, outputs=[capability_notice, submit_edit],
        )
        detect_button.click(
            capabilities, outputs=[capability_notice, submit_edit],
        )
        timer = gr.Timer(3)
        timer.tick(update_edit, edit_state, edit_state)
        timer.tick(update_story, story_state, story_state)
    return demo


def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
