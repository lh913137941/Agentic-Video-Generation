# OpenAI 兼容多模态 API：视频理解与自动分镜（2026-10-10）

> MVP 目标：在 Windows + FFmpeg + Gradio 上，通过**独立配置的视觉聊天模型 API**理解上传视频的关键帧，并得到真正结构化的分镜草稿。它不取代视频生成模型，也**不表示已支持上传原片的 V2V 编辑**。

## 1. 用户工作流

**选择视频 →「◉ 视频理解」→ 输入分析要求（可留空）→ 生成分镜草稿 → 检查模型识别结果与分镜 →「确认分镜并生成粗剪」→ 在同一工作台修改单镜头、导出 MP4。**

新技能不需要视频生成 GPU；关键帧由本地 FFmpeg 提取，视觉分析使用云端或本机 OpenAI 兼容 Chat Completions API。粗剪实际生成仍采用原有 Mock 或 Omni 后端；默认 Mock 得到可播放的色块视频而**不是真实 AI 画面**。

## 2. 配置方式

在启动 FastAPI 的**服务端进程环境变量**中设置：

| 变量 | 说明 |
|---|---|
| \`AVG_VISION_URL\` | API 根地址。示例 \`https://api.openai.com\` 或 \`https://服务商/v1\`；自动拼接 \`/v1/chat/completions\` 或 \`/chat/completions\` |
| \`AVG_VISION_MODEL\` | 具体支持图像输入的视觉模型名称。留空就不启用视觉理解 |
| \`AVG_VISION_API_KEY\` | API 密钥，只存服务端进程环境，不透传到浏览器、响应或日志 |
| \`AVG_VISION_FRAMES\` | 采样帧数，默认 4，允许 1–8 |
| \`AVG_VISION_TIMEOUT_SECONDS\` | 远端调用超时，默认 120 秒，允许 10–600 |
| \`AVG_PARSER\` | 旧文本分镜解析：\`rule\` 或 \`llm\`，与本次视觉理解解耦 |
| \`AVG_BACKEND\` | 视频生成：\`mock\` 或 \`omni\`，不由配置视觉模型自动修改 |

Git Bash 例子（把示例值换成真实配置，避免提交密钥到 Git）：

\`\`\`bash
export AVG_VISION_URL="https://api.openai.com"
export AVG_VISION_MODEL="<支持图像的模型ID>"
read -rsp 'Vision API Key: ' AVG_VISION_API_KEY
export AVG_VISION_API_KEY
echo
python -m agentic_video.api
\`\`\`

另一个终端运行 \`python -m agentic_video.ui\`。访问 \`http://127.0.0.1:8000/api/capabilities\` 观察 \`video_understanding\`、\`vision_model\`、\`vision_frames_sent\`。**配置可检测不等于模型/API 兼容性验证已通过**。不要把 API Key 输入 Prompt 或截图发布。

安全边界：远程 API 地址必须是 HTTPS；仅允许 localhost 的 HTTP，用于本地代理。模型服务还必须实现图片 \`image_url\` data URL 的 Chat Completions 输入与文本 JSON 输出；不支持者会得到明确失败而非伪造分镜。

## 3. 实现接口

- \`POST /api/video-understand\`：multipart \`file\`（本机 MP4）、\`instruction\`、\`shot_count\`、\`theme_style\`。服务端读取不超过 100 MiB 的 MP4，校验视频轨与最长 30 秒，然后抽取 1–8 帧低分辨率 JPEG。
- \`GET /api/capabilities\`：新增 \`video_understanding\`（模型是否配置）、\`vision_model\`（模型名）、\`vision_frames_sent\`（预计采样帧数），并保留独立的 \`video_edit_upload\`。任何响应不得出现 API Key。
- 响应为 \`summary\`、\`observations\`、\`sampled_seconds\`、\`project\`，其中 project 是真实持久化 DRAFT 分镜工程，内部镜头由视觉模型给出的 ShotSpec 直接构建，**不经过 RuleStoryboardParser 的再次改写**。
- 用户点击「确认分镜并生成粗剪」后，使用现有 \`POST /api/projects/{id}/generate\`；此动作与云端视觉分析分离，以减少不必要的生成耗时与费用。
- 原有文本剧本入口 \`POST /api/projects\` 与视频编辑 \`POST /api/video-edits\` 保持不变。

## 4. 帧采样与隐私限制

帧采样均匀选取视频中部不同时间点，尺寸最大 512 px，压缩为 JPEG，再以 \`data:image/jpeg;base64,...\` 输入模型。**不将原始 MP4、不传音轨、不上传用户的本地路径**给模型服务。音频/旁白/字幕 OCR、全时间帧准确识别不在本次范围内。模型只观察有限抽样帧，可能错过短暂事件，摘要和分镜均应由用户人工确认。

临时原视频位于系统临时目录，请求结束后自动清理；模型请求只有选中的图片帧和文字指令。视觉服务不可用时，不创建分镜项目，不降级伪造“AI 分析结果”。HTTP 鉴权/网络错误不回显 provider 原始响应文本和密钥。

## 5. 单页 UI 路由

- 当前技能新增「◉ 视频理解」，只在服务端配好视觉模型后允许提交。
- 上传 MP4 仍使用稳定的文件附件及只读播放器；可先用本地 FFmpeg 裁剪。理解任务使用当前实际选择的裁剪后路径。
- 视频理解结果在右侧作品区展示摘要、可见事实、时间戳、分镜描述；下方复用原有分镜卡片。
- 单击生成粗剪后转到「创作分镜」的项目上下文，保留已生成的视觉计划与来源会话记录；即使原视频仍上传，也不把它冒充视频生成模型的原片条件输入。
- 视觉模型 API 与 V2V 模型状态相互独立，配置前者**不会解锁“视频编辑需要 V2V 模型”按钮**。

## 6. 测试与待验证

单测/集成覆盖：（1）未配置返回 409；（2）API Key 不出现在能力 JSON；（3）模拟视觉服务返回模型分镜并创建真实 DRAFT 项目；（4）实际 FFmpeg 抽帧并验证 multipart/json 消息中包含 JPEG data URLs，而非源 MP4；（5）HTTP/JSON 错误安全处理；（6）前端技能切换、能力检测、转草稿、HTML 安全转义与旧 API 兼容性。

尚未在用户自己的云端账号上进行真实 API 连通、成本、配额和模型响应兼容验收；此外跨任务历史持久化、多轮自主 Agent、音频识别和精准视觉时间定位仍未实现。
