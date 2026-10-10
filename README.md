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

Open `http://127.0.0.1:7860` for the **single-page conversational creator workspace**. Upload an MP4 and explicitly choose **视频编辑** (requires a real V2V-capable backend), or type a script and choose **创作分镜** (works in Mock mode). The same page shows status events, video comparison, rough-cut preview and targeted shot regeneration. Uploaded MP4 files are never silently ignored by the storyboard workflow. API schema: `http://127.0.0.1:8000/docs`.

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
