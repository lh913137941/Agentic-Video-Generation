"""Minimal standalone P0 Gradio studio; does not replace existing V2V/storyboard UI.

Run API first, then: agentic-video-mvp. All state lives in the API service.
"""
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx


def build_ui(api_url: str | None = None):
    import gradio as gr

    base = (api_url or os.getenv("AVG_API_URL", "http://127.0.0.1:8000")).rstrip("/")

    def send(method: str, path: str, *, payload=None, files=None):
        try:
            with httpx.Client(timeout=1200) as client:
                result = client.request(method, base + path, json=payload, files=files)
            if result.status_code >= 400:
                try:
                    message = result.json().get("detail", result.text)
                except ValueError:
                    message = result.text
                raise gr.Error(f"API HTTP {result.status_code}: {str(message)[:500]}")
            return result.json()
        except httpx.HTTPError as exc:
            raise gr.Error(f"Cannot reach editing API: {exc}") from exc

    def upload_video(source):
        if not source:
            raise gr.Error("请先上传 MP4")
        path = Path(source)
        with path.open("rb") as handle:
            record = send("POST", "/api/mvp/projects", files={
                "file": (path.name, handle, "video/mp4"),
            })
        return record["id"], json.dumps(record, ensure_ascii=False, indent=2), "已导入视频；下一步语音识别"

    def transcribe(project_id):
        if not project_id:
            raise gr.Error("请先上传视频")
        record = send("POST", f"/api/mvp/projects/{project_id}/transcribe")
        return json.dumps(record["transcript"], ensure_ascii=False, indent=2), "ASR 已完成"

    def save_transcript(project_id, text):
        if not project_id:
            raise gr.Error("请先上传视频")
        try:
            segments = json.loads(text)
        except ValueError as exc:
            raise gr.Error("转写文本必须是包含时间戳的 JSON 数组") from exc
        record = send("PUT", f"/api/mvp/projects/{project_id}/transcript",
                      payload={"segments": segments})
        return json.dumps(record["transcript"], ensure_ascii=False, indent=2), "已保存转写文本"

    def plan(project_id, prompt, duration):
        if not project_id:
            raise gr.Error("请先上传视频")
        result = send("POST", f"/api/mvp/projects/{project_id}/plans", payload={
            "instruction": prompt,
            "target_seconds": int(duration),
            "captions": True,
        })
        return result["version"], json.dumps(result, ensure_ascii=False, indent=2), "草案已生成，等待人工审阅"

    def revise(project_id, version, plan_json):
        try:
            plan_obj = json.loads(plan_json)
        except ValueError as exc:
            raise gr.Error("EditPlan JSON 无效") from exc
        version = int(version)
        result = send("PATCH", f"/api/mvp/projects/{project_id}/plans/{version}", payload={
            "clips": plan_obj["clips"],
            "captions": plan_obj["captions"],
            "aspect": plan_obj["aspect"],
        })
        return result["version"], json.dumps(result, ensure_ascii=False, indent=2), "已创建修订版本；需要重新确认"

    def approve(project_id, version):
        result = send("POST", f"/api/mvp/projects/{project_id}/plans/{int(version)}/approve")
        return json.dumps(result, ensure_ascii=False, indent=2), "方案已人工确认，可以导出"

    def start_render(project_id, version):
        r = send("POST", f"/api/mvp/projects/{project_id}/renders?version={int(version)}")
        return r["id"], json.dumps(r, ensure_ascii=False, indent=2)

    def poll(project_id, job_id):
        if not job_id:
            raise gr.Error("请先提交导出任务")
        job = send("GET", f"/api/mvp/projects/{project_id}/renders/{job_id}")
        if job["status"] != "READY":
            return json.dumps(job, ensure_ascii=False, indent=2), None
        url = f"{base}/api/mvp/projects/{project_id}/renders/{job_id}/file"
        path = Path(tempfile.gettempdir()) / ("avg-p0-" + uuid4().hex + ".mp4")
        try:
            with httpx.Client(timeout=120) as client:
                with client.stream("GET", url) as result:
                    result.raise_for_status()
                    with path.open("wb") as out:
                        for chunk in result.iter_bytes():
                            out.write(chunk)
        except httpx.HTTPError as exc:
            path.unlink(missing_ok=True)
            raise gr.Error("下载验证后的视频失败") from exc
        return json.dumps(job, ensure_ascii=False, indent=2), str(path)

    with gr.Blocks(title="Agentic Video · P0 视频智能剪辑") as app:
        gr.Markdown("# P0 · 长视频 → 短视频剪辑（可审阅时间线）")
        gr.Markdown("导入 → ASR / 人工转写 → 规则草案 → 修改 → 明确批准 → 导出。当前不调用 V2V，也不将规则规划伪装成 LLM。")
        project_id = gr.State("")
        version = gr.State(0)
        job_id = gr.State("")
        with gr.Row():
            source = gr.File(label="视频素材 MP4", file_types=[".mp4"], type="filepath")
            with gr.Column():
                upload_btn = gr.Button("1. 上传素材", variant="primary")
                project_json = gr.JSON(label="素材元信息")
                notice = gr.Textbox(label="操作状态")
        with gr.Row():
            transcript = gr.Code(label="转写 JSON（毫秒时间戳，可编辑）", language="json",
                                 value="[]", lines=12)
            with gr.Column():
                asr_btn = gr.Button("2. 本地 ASR 转写（需安装 faster-whisper）")
                manual_btn = gr.Button("保存人工转写")
                instruction = gr.Textbox(label="剪辑要求", value="做成45秒9:16竖屏短视频，去掉停顿，加中文字幕")
                target = gr.Slider(5, 90, value=45, step=1, label="目标时长 / 秒")
                plan_btn = gr.Button("3. 生成可审阅剪辑草案")
        edit_json = gr.Code(label="EditPlan（修改 clips 来源时间码、字幕和画幅后保存）",
                            language="json", lines=18)
        with gr.Row():
            save_btn = gr.Button("4. 保存修改版本")
            approve_btn = gr.Button("5. 批准当前版本")
            render_btn = gr.Button("6. 导出 MP4", variant="primary")
            poll_btn = gr.Button("刷新导出状态")
        render_state = gr.Code(label="导出任务状态", language="json", lines=6)
        result_video = gr.Video(label="可播放的最终视频")
        upload_btn.click(upload_video, [source], [project_id, project_json, notice])
        asr_btn.click(transcribe, [project_id], [transcript, notice])
        manual_btn.click(save_transcript, [project_id, transcript], [transcript, notice])
        plan_btn.click(plan, [project_id, instruction, target], [version, edit_json, notice])
        save_btn.click(revise, [project_id, version, edit_json], [version, edit_json, notice])
        approve_btn.click(approve, [project_id, version], [edit_json, notice])
        render_btn.click(start_render, [project_id, version], [job_id, render_state])
        poll_btn.click(poll, [project_id, job_id], [render_state, result_video])
    return app


def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
