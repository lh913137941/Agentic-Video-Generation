# Agentic Video Generation

A storyboard-first, extensible **script → shots → generated clips → rough-cut MP4** demo. The first working milestone uses **FastAPI + asyncio + Gradio + FFmpeg**; vLLM-Omni is an optional remote inference backend.

## Quick start (no GPU)

Prerequisites: Python 3.10+ and `ffmpeg` on PATH.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ui,test]"

# Terminal A
agentic-video-api

# Terminal B
agentic-video-ui
```

Open `http://127.0.0.1:7860` for the **single-page conversational creator workspace**. Uploads use a compact file attachment and preview in a dark conversational composer. Click **✎ 高级编辑 · 标记画面** to open an immersive dark editor: draw on the enlarged video frame, describe the desired change and **保存并返回创作**. Detailed timeline/START-END frame thumbnails, FFmpeg trim and local stationary-region blur remain available in the editor's optional advanced section. Area-replacement AI editing still requires a compatible mask-aware V2V model; a saved mark alone does not perform semantic inpainting. Upload an MP4 and explicitly choose **视频编辑** (requires a real V2V-capable backend), or type a script and choose **创作分镜** (works in Mock mode). The same page shows status events, video comparison, rough-cut preview and targeted shot regeneration. Uploaded MP4 files are never silently ignored by the storyboard workflow. API schema: `http://127.0.0.1:8000/docs`.

Mock mode (`AVG_BACKEND=mock`) is the default and creates playable silent color clips using FFmpeg; it does **not** generate AI visuals.

### API quick smoke test

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s -X POST http://127.0.0.1:8000/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"script":"A sunrise. A cup of coffee. A smiling barista.","shot_count":3}'
```

Copy `project_id` from the response:

```bash
curl -s -X POST http://127.0.0.1:8000/api/projects/PROJECT_ID/generate
curl -s http://127.0.0.1:8000/api/projects/PROJECT_ID
curl -L http://127.0.0.1:8000/api/projects/PROJECT_ID/video -o rough-cut.mp4
```

Generation is asynchronous. Wait until project `status` is `READY` before downloading.


## V2V video editing API configuration

Open **⚙ V2V 模型 API 配置** under the main creator composer.

Choose Mock (no AI editing), vLLM-Omni (local video editing endpoint), or **fal.ai Wan 2.7 Edit Video** (cloud video editing using MP4 + prompt). The cloud option requires a provider API key and a 2–10 second MP4 and may incur inference charges. This is separate from AVG_VISION_* (OpenAI-compatible vision understanding).

Install the cloud adapter dependency:

```bash
python -m pip install -e ".[ui,test,v2v]"
```

Set provider and API key in the UI password field, choose **保存 V2V 配置**, then **restart the FastAPI backend** and press **重新检测 API 生效状态**. This is local single-user development. Secrets are stored in ~/.agentic_video/v2v.json under the OS account, not committed to Git or served by the public capabilities endpoint. On Windows protect your user profile from other users. Never publish the UI/API ports to the Internet without authentication.

Environment variables take priority if explicitly set when starting the API:

```bash
export AVG_BACKEND=fal
export AVG_FAL_MODEL=fal-ai/wan/v2.7/edit-video
read -rsp "fal.ai Key: " FAL_KEY; export FAL_KEY; echo
python -m agentic_video.api
```

Local Omni instead uses AVG_BACKEND=omni and AVG_OMNI_URL=http://127.0.0.1:8091. Check /api/capabilities for video_edit_backend, v2v_client_installed and max allowed video duration. Saved settings only apply after API restart; API Key never appears in capabilities. With fal.ai the original MP4 is uploaded to its provider storage, so review provider privacy and billing terms before running it. Saved frame drawings/JSON are not passed as masks to this cloud adapter. It only supplies the video and text instruction.

## OpenAI-compatible multimodal video understanding

You can connect a vision-capable Chat Completions model **without deploying a video-generation GPU**. The backend samples MP4 frames locally with FFmpeg, sends a small number of JPEG images to a configured API, validates the structured shot plan, and creates a DRAFT storyboard project. The UI adds a distinct **「◉ 视频理解」** skill. Review the proposed shots before clicking "确认分镜并生成粗剪"; the latter uses the existing Mock/Omni video backend.

```bash
export AVG_VISION_URL=https://api.openai.com
export AVG_VISION_MODEL="<vision-capable-model-id>"
read -rsp "Vision API key: " AVG_VISION_API_KEY; export AVG_VISION_API_KEY; echo
python -m agentic_video.api
```

The compatible provider must accept `image_url` JPEG data URLs at `/v1/chat/completions` and respond with plain JSON text. `AVG_VISION_MODEL` is separate from `AVG_LLM_MODEL` (text storyboard parser) and `AVG_BACKEND` (Mock/Omni video generation). Configuring vision **does not enable V2V editing**. No audio is transmitted or understood. See [multimodal API integration guide](docs/multimodal_api.md).

## Real vLLM-Omni inference

Deploy a compatible model separately (e.g. `vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091`) and set:

```bash
export AVG_BACKEND=omni
export AVG_OMNI_URL=http://127.0.0.1:8091
# Must match your model's supported width / height / fps / frame count:
export AVG_WIDTH=832 AVG_HEIGHT=480 AVG_MODEL_FPS=16
export AVG_MODEL_FRAME_STRIDE=4  # Wan-style frame count 4k+1; set 0 for models without this rule
agentic-video-api
```

For the upload-first editor, select **a V2V-capable model** and upload a valid MP4 (maximum 30 seconds and 100 MiB). The editor sends an `input_reference` file with the prompt; not every vLLM-Omni model supports V2V. Use `GET /api/capabilities` to check UI/backend mode. In Mock mode, this editor is deliberately unavailable rather than silently fabricating an edit. See [unified creator workspace design](docs/unified_workspace_design.md) and [original UI guide mapping](docs/ui_workflow.md).

The adapter **POSTs multipart** to `/v1/videos`, polls `GET /v1/videos/{id}`, and downloads `/content`. The chosen model must accept the configured frame-count sequence and resolution. No GPU end-to-end validation is claimed in this repository scaffold.

### Environment & project structure

See [`.env.example`](.env.example) for configuration. `AVG_DATA_DIR` defaults to `./data` and must not be shared by multiple API worker processes.

```text
src/agentic_video/
  api.py         HTTP endpoints and dependency assembly
  ui.py          Single-page conversational creator composer
  workspace.py   Frontend session state and explicit workflow routing
  assets/        CSS theme
  service.py     Job/state/revision orchestration
  models.py      Pydantic contracts
  parsers.py     Rule-based and LLM storyboard parsers
  backends.py    Mock and vLLM-Omni video generators
  composer.py    FFmpeg normalized concat
  store.py       Atomic local JSON persistence
  config.py      Environment settings
tests/           Unit + FFmpeg integration tests
docs/            Expanded implementation design
```

```bash
pip install -e ".[test,ui]"
pytest -q
```

## Scope

Current MVP delivers an upload-first V2V editing interface (model support unverified), storyboard editing, targeted regeneration, state/error reporting, silent normalized rough-cut composition and MP4 download. Audio/TTS, transitions, reference-driven consistency, B-Roll planning, retries with backoff and distributed workers are future extensions. For architecture, interface contracts, failure handling and milestone gates, read [优化版设计文档](docs/vllm_omni_demo.md).

Development branch: `feature/extensible-video-mvp-20261010`.
