"""Two-workflow creator UI: upload-first smart edit + storyboard rough-cut.

The edit workspace follows the six-step usability pattern in the 2026-10-10
Jimeng quick-experience guide. This does not embed or use Jimeng services.
"""
import html
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

STUDIO_CSS = """
.gradio-container { max-width: 1440px !important; margin: auto !important; }
.studio-hero {
    background: linear-gradient(125deg, #17182a 0%, #292043 53%, #49306c 100%);
    border: 1px solid rgba(180,170,255,.19); border-radius: 22px;
    padding: 28px 34px; color: #fff; margin-bottom: 16px;
}
.studio-hero h1 { margin: 0 0 9px; color: #fff; font-size: 30px; font-weight: 730; }
.studio-hero p { margin: 0; opacity: .83; font-size: 14px; }
.studio-kicker { color: #e4ccff; letter-spacing: .15em; font-size: 11px;
    font-weight: 700; margin-bottom: 10px; }
.workspace-caption { font-size: 13px; color: var(--body-text-color-subdued); }
.studio-note { border-left: 3px solid #9270ec; padding-left: 12px; }
"""


def choose_edit_preset(name: str) -> str:
    return EDIT_PRESETS.get(name, "")


def video_edit_instruction(prompt: str) -> str:
    """Do not silently rewrite preservation constraints or claim exact masking."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("请先输入编辑指令")
    if len(prompt) > 3000:
        raise ValueError("编辑指令过长（最多 3000 字符）")
    return prompt


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
                resp = client.get(base + "/api/capabilities")
                resp.raise_for_status()
                info = resp.json()
            if info.get("video_edit_upload"):
                return (
                    "**智能编辑推理通道：已配置 vLLM-Omni。** 请确认服务端"
                    "实际模型支持 Video-to-Video 输入；是否保持原人物与动作"
                    "取决于模型能力，不能由界面保证。"
                )
            return (
                "**当前运行于 Mock / 无视频编辑后端。** 你可以上传、预览视频并"
                "准备编辑指令；点击「生成编辑结果」时后端会明确拒绝，而不会"
                "将一段新生成的视频冒充为原片编辑。启用需设置 "
                "\`AVG_BACKEND=omni\` 并部署支持 V2V 的模型。"
            )
        except (httpx.HTTPError, ValueError):
            return "**API 尚未连通。** 请先启动 \`agentic-video-api\`，再刷新页面。"

    def start_edit(source: str | None, prompt: str, existing: dict | None):
        if existing and existing.get("status") in ("QUEUED", "GENERATING"):
            raise gr.Error("当前编辑任务未结束，请先完成或等待本次任务")
        if not source:
            raise gr.Error("第 1 步：先上传本地 MP4 原视频")
        try:
            valid_prompt = video_edit_instruction(prompt)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        path = Path(source)
        if path.suffix.lower() != ".mp4":
            raise gr.Error("仅支持上传 MP4 视频")
        if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
            raise gr.Error("视频不存在或超过 100 MiB")
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

    def start_story(project):
        if not project:
            raise gr.Error("先解析分镜，再生成视频")
        return send("POST", f"/api/projects/{project['project_id']}/generate")

    def update_story(project):
        if not project or project.get("status") not in {
            "DRAFT", "GENERATING", "STITCHING"
        }:
            return gr.skip()
        try:
            with httpx.Client(timeout=8) as client:
                response = client.get(base + f"/api/projects/{project['project_id']}")
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
                "prompt": prompt, "duration": duration,
                "camera_motion": motion,
            })
            return send(
                "POST", f"/api/projects/{pid}/shots/{shot_id}/regenerate"
            )
        return handler

    with gr.Blocks(
        title="Agentic Video Studio",
        theme=gr.themes.Soft(primary_hue="violet"),
        css=STUDIO_CSS,
    ) as demo:
        gr.HTML(
            '<section class="studio-hero">'
            '<div class="studio-kicker">AGENTIC VIDEO STUDIO · MVP</div>'
            '<h1>把想法变成可编辑的视频</h1>'
            '<p>从已有视频定向修改，或从剧本生成分镜粗剪。'
            '两种工作流独立，不会混淆视频编辑与文生视频。</p>'
            '</section>'
        )

        with gr.Tabs():
            with gr.Tab("✦ 智能编辑 · 已有视频", id="edit"):
                gr.Markdown(
                    "上传原片 → 选择示例或输入修改指令 → 生成 → "
                    "查看原片与结果 → 下载。建议首次使用 5～10 秒 MP4。"
                )
                capability_notice = gr.Markdown(
                    "正在检查后端能力…", elem_classes=["studio-note"]
                )
                edit_state = gr.State({})
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        gr.Markdown("### 01 · 上传原视频")
                        original = gr.Video(
                            label="原视频（MP4，最长 30 秒 / 最大 100 MiB）",
                            sources=["upload"], format="mp4",
                            interactive=True,
                        )
                        gr.Markdown("### 02 · 告诉 AI 需要修改什么")
                        preset = gr.Dropdown(
                            choices=list(EDIT_PRESETS),
                            value="白天变黄昏 · 推荐首测",
                            label="编辑指令模板",
                        )
                        use_preset = gr.Button("使用此模板", variant="secondary")
                        instruction = gr.Textbox(
                            label="编辑指令 / Prompt",
                            value=EDIT_PRESETS["白天变黄昏 · 推荐首测"],
                            lines=6,
                            max_lines=12,
                            placeholder="只改变……；保持人物、动作、镜头、时长不变",
                        )
                        use_preset.click(
                            choose_edit_preset, preset, instruction
                        )
                        gr.Markdown(
                            "建议一次只修改一类元素，明确写出**必须保持不变**"
                            "的内容。模型可能无法严格保证原片身份、运动和时序。"
                        )
                        with gr.Accordion("高级编辑 · 区域框选与时间定位（规划中）",
                                          open=False):
                            gr.Markdown(
                                "参考体验指南，未来可以对某一帧选择衣服等区域，"
                                "或指定目标时间。当前后端没有区域 Mask 或"
                                "精确时间段条件的通用实现，因此暂不提供"
                                "容易产生误解的假控件。"
                            )
                        submit_edit = gr.Button(
                            "03 · 生成编辑结果", variant="primary", size="lg"
                        )
                        gr.Markdown(
                            "提交前请确认模型支持 Video-to-Video；生成耗时、"
                            "显存与外部服务费用由所部署的模型决定，"
                            "本系统不提供积分或费用估算。"
                        )
                    with gr.Column(scale=7):
                        gr.Markdown("### 04 · 原片与生成结果")
                        @gr.render(inputs=edit_state)
                        def edit_preview(job):
                            if not job:
                                gr.Markdown(
                                    "预览原片后提交编辑任务，结果将显示在这里。"
                                    "完成后可对照人物、光照、动作及镜头节奏。"
                                )
                                return
                            job_id = job["job_id"]
                            status = job["status"]
                            error = html.escape(str(job.get("error") or ""))
                            gr.Markdown(
                                f"**任务状态：{status}** · ID: \`{job_id}\`"
                                + (f" · **错误：{error}**" if error else "")
                            )
                            with gr.Row():
                                with gr.Column():
                                    gr.Video(
                                        value=base + f"/api/video-edits/{job_id}/source",
                                        label="原始视频", interactive=False,
                                    )
                                with gr.Column():
                                    if status == "READY":
                                        gr.Video(
                                            value=base + f"/api/video-edits/{job_id}/result",
                                            label="编辑结果", interactive=False,
                                        )
                                    else:
                                        gr.Markdown(
                                            "编辑结果尚未就绪。"
                                            if status != "FAILED"
                                            else "生成失败，请按错误提示检查模型能力或修改指令。"
                                        )
                            if status == "READY":
                                gr.Markdown(
                                    f"[↓ 下载编辑后的 MP4]"
                                    f"({base}/api/video-edits/{job_id}/result?download=1)"
                                )
                            gr.Markdown(
                                "对照检查：人物身份、动作、镜头运动、"
                                "开头/中间/结尾的位置与实际视频时长。"
                            )

            with gr.Tab("▦ 剧本成片 · 分镜工作流", id="storyboard"):
                story_state = gr.State({})
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.Markdown("### 01 · 剧本文案")
                        script = gr.Textbox(
                            label="剧本 / 口播文案",
                            lines=7,
                            placeholder="例如：清晨，咖啡师打开咖啡馆。"
                            "热气从杯中升起。顾客接过咖啡微笑。",
                        )
                        style = gr.Textbox(
                            label="全片统一视觉风格",
                            value="cinematic, consistent color palette",
                        )
                        shot_count = gr.Slider(
                            1, 5, step=1, value=3, label="分镜数量"
                        )
                        parse_button = gr.Button("02 · 解析分镜", variant="primary")
                        generate_button = gr.Button("03 · 生成并合成视频")
                        refresh_button = gr.Button("刷新项目状态", variant="secondary")
                    with gr.Column(scale=7):
                        gr.Markdown("### 粗剪预览与导出")
                        @gr.render(inputs=story_state)
                        def final_preview(project):
                            if not project:
                                gr.Markdown("解析分镜后，这里展示项目进度及粗剪。")
                                return
                            status = project["status"]
                            err = html.escape(str(project.get("error") or ""))
                            gr.Markdown(
                                f"**项目状态：{status}** · "
                                f"工程 \`{project['project_id']}\`"
                                + (f" · **错误：{err}**" if err else "")
                            )
                            if status == "READY":
                                url = base + f"/api/projects/{project['project_id']}/video"
                                gr.Video(
                                    value=url, label="当前粗剪 MP4",
                                    interactive=False,
                                )
                                gr.Markdown(f"[↓ 下载粗剪视频]({url}?download=1)")
                            else:
                                gr.Markdown(
                                    "镜头生成后会自动拼接。编辑分镜后，"
                                    "旧粗剪不会作为当前版本显示。"
                                )
                gr.Markdown("### 04 · 分镜卡片 / 局部重投")
                gr.Markdown(
                    "每张卡片对应一个独立镜头。调整 Prompt、时长或运镜后，"
                    "点击「保存修改并重投」，其它已就绪的分镜将继续复用。"
                )

                @gr.render(inputs=story_state)
                def shot_cards(project):
                    if not project:
                        return
                    for shot in project["shots"]:
                        with gr.Group():
                            gr.Markdown(
                                f"#### 镜头 {shot['id']:02d} · {shot['status']}"
                                f" · 版本 {shot['revision']}"
                                f"/{shot['rendered_revision']}"
                            )
                            with gr.Row():
                                with gr.Column(scale=6):
                                    shot_prompt = gr.Textbox(
                                        value=shot["prompt"],
                                        label=f"镜头 {shot['id']} 画面描述",
                                        lines=3,
                                        key=f"shot-prompt-{shot['id']}",
                                    )
                                    with gr.Row():
                                        duration = gr.Slider(
                                            1, 10, value=shot["duration"],
                                            step=0.5, label="时长（秒）",
                                            key=f"shot-duration-{shot['id']}",
                                        )
                                        motion = gr.Dropdown(
                                            ["static", "pan", "tilt", "tracking",
                                             "dolly", "handheld"],
                                            value=shot["camera_motion"]
                                            if shot["camera_motion"] in
                                            {"static", "pan", "tilt", "tracking",
                                             "dolly", "handheld"} else "static",
                                            label="运镜描述",
                                            key=f"shot-motion-{shot['id']}",
                                        )
                                    rerun = gr.Button(
                                        f"↻ 保存修改并重投镜头 {shot['id']}",
                                        key=f"shot-rerun-{shot['id']}",
                                    )
                                    rerun.click(
                                        retry_shot(shot["id"]),
                                        [story_state, shot_prompt, duration, motion],
                                        story_state,
                                    )
                                with gr.Column(scale=5):
                                    if (
                                        shot["status"] == "READY" and
                                        shot["rendered_revision"] == shot["revision"] and
                                        shot["video_path"]
                                    ):
                                        gr.Video(
                                            value=base + f"/api/projects/"
                                            f"{project['project_id']}/shots/"
                                            f"{shot['id']}/video",
                                            label="镜头预览", interactive=False,
                                        )
                                    else:
                                        gr.Markdown(
                                            f"镜头当前状态：**{shot['status']}**"
                                            + (
                                                f" · {html.escape(shot['error'])}"
                                                if shot.get("error") else ""
                                            )
                                        )
                parse_button.click(
                    create_story, [script, style, shot_count], story_state
                )
                generate_button.click(start_story, story_state, story_state)
                refresh_button.click(update_story, story_state, story_state)

        gr.Markdown(
            "本产品是独立的视频生成工作台，并非即梦 AI 服务。"
            "当前高级区域/时间编辑尚未接入；"
            "Mock 模式只用于分镜生成流程测试，不提供真正的 AI 视频编辑。",
            elem_classes=["workspace-caption"],
        )
        submit_edit.click(
            start_edit, [original, instruction, edit_state], edit_state
        )
        demo.load(capabilities, outputs=capability_notice)
        timer = gr.Timer(3)
        timer.tick(update_edit, edit_state, edit_state)
        timer.tick(update_story, story_state, story_state)

    return demo


def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
