# P0 口播剪辑 MVP：真实媒体闭环（2026-10-12）

依据 Library 的 `Video_Agent_MVP.docx` 第 3–5、8–10 章。目标是 **上传现有口播视频 → ASR（或人工转写）→ 可审阅来源时间码的 EditPlan → 人工修改/批准 → FFmpeg 输出 9:16 MP4（保留音频、可选硬字幕）**。

本分支基于 `feature/extensible-video-mvp-20261010`，不修改旧分镜生成或 Omni/fal V2V 路径。

## 快速运行（Windows / Linux / macOS）

需要 Python 3.10+ 和 PATH 中可用的 `ffmpeg`、`ffprobe`；视频渲染还需要 FFmpeg 包含 `libx264`、`aac` 以及硬字幕时的 `subtitles/libass` 滤镜。

```bash
python -m pip install -e ".[ui,test,asr]"
# 本地 ASR 默认 faster-whisper small / CPU int8；首次运行需要下载权重。
# 可选：AVG_ASR_MODEL=base 或 AVG_ASR_MODEL=small
agentic-video-api
# 新终端
agentic-video-mvp
```

打开 `http://127.0.0.1:7861`；FastAPI Swagger 在 `http://127.0.0.1:8000/docs`。

ASR 不是必选依赖：离线环境不能下载模型时，用户可直接上传审核过的时间戳转写 JSON，绝不伪造识别结果。没有 ASR、又没有人工转写时，规划会明确失败。

### P0 的两个 Planner

- **规则草案**：无需 LLM，按语句顺序选择内容、去除空档，标记 `planner=rules`；只用于 G1 / 应急回退，不声称自动识别“精彩内容”。
- **LLM 语义选段**：可选，设置 `AVG_P0_LLM_MODEL` 和 `AVG_P0_LLM_URL`（OpenAI-compatible 的 HTTPS / localhost 地址），可选 `AVG_P0_LLM_API_KEY`。调用 `POST /api/mvp/projects/{id}/plans/agent`。服务端只发送有时间戳的**转写文本**，LLM 返回实际 segment ID，严格校验合法性并由服务端生成时间码、字幕；此阶段不发送原始 MP4。超过 300 段的长转写需人工分段，暂不支持全文语义检索。

**隐私**：外部 LLM 将接收转写内容。敏感视频请使用本地兼容模型或规则 Planner；密钥必须只在服务端设置。

## 核心 API

| 方法 | 路由 | 作用 |
|---|---|---|
| POST | `/api/mvp/projects` | 上传 MP4，存储源媒体并用 ffprobe 读取时长/音轨 |
| POST | `/api/mvp/projects/{id}/transcribe` | 可选 faster-whisper ASR（目前同步耗时调用） |
| PUT | `/api/mvp/projects/{id}/transcript` | 提供/修改带毫秒时间戳的人工转写 |
| POST | `/api/mvp/projects/{id}/plans` | 规则版 EditPlan 草案 |
| POST | `/api/mvp/projects/{id}/plans/agent` | 受约束的 LLM 语义选段草案 |
| GET | `/api/mvp/projects/{id}/plans/{version}` | 查询指定计划版本 |
| PATCH | `/api/mvp/projects/{id}/plans/{version}` | 修改 clips、字幕、画幅，产生新版本 |
| POST | `/api/mvp/projects/{id}/plans/{version}/approve` | 人工确认最新版本 |
| POST | `/api/mvp/projects/{id}/renders?version=N` | 提交渲染；同一成功版本幂等 |
| GET | `/api/mvp/projects/{id}/renders/{render_id}` | 轮询状态及指标 |
| GET | `/api/mvp/projects/{id}/renders/{render_id}/file` | VERIFIED READY MP4 预览或下载 |

## 最小数据契约

```json
{
  "schema_version": 1,
  "version": 1,
  "instruction": "做成竖屏短视频并加字幕",
  "aspect": "9:16",
  "planner": "rules",
  "approval": "pending",
  "clips": [
    {"source_id":"source","in_ms":1000,"out_ms":3600,"reason":"ASR 语句选段；请人工确认重点内容"}
  ],
  "captions": [
    {"start_ms":0,"end_ms":2600,"text":"字幕内容"}
  ]
}
```

**cl​ips 的 `in_ms/out_ms` 是源视频时钟；captions 的 `start_ms/end_ms` 是剪辑后输出时钟。** 统一为整数毫秒。时间码越界、非法顺序、空片段、未批准渲染均会拒绝。修改创建新版本，不更改原始视频。

### API 人工转写示例

```bash
curl -X PUT http://127.0.0.1:8000/api/mvp/projects/PROJECT_ID/transcript \
  -H "Content-Type: application/json" \
  -d '{"segments":[{"start_ms":1000,"end_ms":3500,"text":"欢迎来到今天的视频"}]}'
```

测试镜头、字幕、9:16 的端到端行为：

```bash
python -m pytest -q tests/test_p0_editing.py
```

CI 使用 FFmpeg 真实生成带音频的 MP4，并对最终视频再次 `ffprobe`，校验时长、画幅及音轨；不是仅用 Mock 将状态标为 READY。

## 参考及工程取舍

- [Pireel](https://github.com/pireel/pireel)：参考其 **Provider 边界、编辑工具与计划分离**；没有复制其 AGPL 代码。
- [OpenChatCut](https://github.com/0xsline/OpenChatCut)：参考其 **时间轴操作可追踪、可撤销、Agent 与 UI 共享状态**；没有复制其 AGPL 代码。
- [agamm/video-agent](https://github.com/agamm/video-agent)：参考源时间码和审核式 EDL。

这一版是**可用的纵向 MVP 切片**，并非全部 15 天目标。已提供完整后台操作和独立轻量 Gradio 页面；但暂未完成高精度可拖拽多轨时间轴、精彩片段质量评分、字幕字体模板、多候选片、自动长静音 VAD 精切、长 ASR 后台任务/自动重试、精细人物自动居中（当前使用居中裁切）。这些在 P0 验收中必须明确列为未完成，不能作为已验证能力宣传。

部署限制：首发单进程、本机可信使用；1 GiB MP4 上限、2 小时媒体上限、最长 120 秒输出、每个任务最多 60 clips。商业部署前必须补鉴权、用户隔离、数据保留、GPU/CPU 配额及持久作业队列。
