"""Lightweight Gradio storyboard cards, backed by the FastAPI service."""
import os
import httpx


def build_ui(api_url: str | None = None):
    import gradio as gr

    base = (api_url or os.getenv("AVG_API_URL", "http://127.0.0.1:8000")).rstrip("/")

    def request(method: str, path: str, payload=None):
        with httpx.Client(timeout=120) as client:
            response = client.request(method, base + path, json=payload)
            response.raise_for_status()
            return response.json()

    def create(script: str, style: str, count: int):
        project = request("POST", "/api/projects", {
            "script": script, "theme_style": style, "shot_count": int(count),
        })
        return project

    def generate(project):
        if not project:
            raise gr.Error("Create a storyboard first")
        return request("POST", f"/api/projects/{project['project_id']}/generate")

    def refresh(project):
        if not project:
            return project
        try:
            return request("GET", f"/api/projects/{project['project_id']}")
        except httpx.HTTPError:
            return project

    def regenerate(shot_id):
        def handler(project, prompt):
            pid = project["project_id"]
            request("PATCH", f"/api/projects/{pid}/shots/{shot_id}", {
                "prompt": prompt,
            })
            return request(
                "POST", f"/api/projects/{pid}/shots/{shot_id}/regenerate"
            )
        return handler

    with gr.Blocks(title="Agentic Video Generation") as demo:
        gr.Markdown("# Agentic Video Generation\n剧本 → 分镜 → 镜头生成 → 粗剪")
        state = gr.State({})
        with gr.Row():
            with gr.Column(scale=2):
                script = gr.Textbox(label="剧本 / 口播文案", lines=6)
                style = gr.Textbox(label="统一视觉风格",
                                   value="cinematic, consistent color palette")
                shot_count = gr.Slider(1, 5, value=3, step=1, label="分镜数量")
                create_button = gr.Button("1. 解析分镜", variant="primary")
                generate_button = gr.Button("2. 生成 / 合成")
                refresh_button = gr.Button("刷新任务状态")
            with gr.Column(scale=3):
                gr.Markdown("### 最终粗剪与项目状态")
                @gr.render(inputs=state)
                def final_panel(project):
                    if not project:
                        gr.Markdown("创建项目后在此预览成片。")
                        return
                    gr.Markdown(
                        f"状态：**{project['status']}** · 项目："
                        f"{project['project_id']} "
                        + (f"· 错误：{project['error']}" if project.get("error") else "")
                    )
                    if project["status"] == "READY":
                        gr.Video(
                            value=base + f"/api/projects/{project['project_id']}/video",
                            label="成片预览", interactive=False,
                        )
        gr.Markdown("### 分镜卡片（支持独立修改与重新生成）")

        @gr.render(inputs=state)
        def cards(project):
            if not project:
                return
            for shot in project["shots"]:
                with gr.Group():
                    gr.Markdown(
                        f"**镜头 {shot['id']}** · {shot['status']} "
                        f"· revision {shot['revision']}/"
                        f"{shot['rendered_revision']}"
                    )
                    prompt = gr.Textbox(
                        value=shot["prompt"], label=f"镜头 {shot['id']} Prompt",
                        lines=2, key=f"prompt-{shot['id']}",
                    )
                    rerun = gr.Button(
                        f"🔄 重新生成镜头 {shot['id']}",
                        key=f"regenerate-{shot['id']}",
                    )
                    rerun.click(regenerate(shot["id"]), [state, prompt], state)
                    if (
                        shot["status"] == "READY"
                        and shot["rendered_revision"] == shot["revision"]
                        and shot["video_path"]
                    ):
                        gr.Video(
                            value=base + f"/api/projects/{project['project_id']}"
                                         f"/shots/{shot['id']}/video",
                            label=f"镜头 {shot['id']} 预览",
                            interactive=False,
                        )
        create_button.click(create, [script, style, shot_count], state)
        generate_button.click(generate, state, state)
        refresh_button.click(refresh, state, state)
        timer = gr.Timer(3)
        timer.tick(refresh, state, state)
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
