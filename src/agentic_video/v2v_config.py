"""Single-user local V2V provider settings.

Never expose API keys through /api/capabilities or Gradio state. The UI and API
are designed to run on the same user's workstation (127.0.0.1).
Restart the API after saving provider changes.
"""
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

FAL_EDIT_MODEL = "fal-ai/wan/v2.7/edit-video"
CONFIG_FILE = Path.home() / ".agentic_video" / "v2v.json"


def _validate(provider: str, omni_url: str, model: str, api_key: str) -> dict:
    if provider not in {"mock", "omni", "fal"}:
        raise ValueError("仅支持 Mock、vLLM-Omni 或 fal.ai 视频编辑")
    if provider == "omni":
        parsed = urlsplit(omni_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or (
            parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise ValueError("Omni URL 必须是有效的 HTTP(S) 服务地址")
        # Public services must use TLS; local development can use HTTP.
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("远程 Omni 服务需要 HTTPS，本机可使用 HTTP")
    if provider == "fal":
        if model != FAL_EDIT_MODEL:
            raise ValueError("当前仅支持已适配的 fal.ai Wan 2.7 Edit Video 模型")
        if not api_key.strip() or len(api_key) > 1024:
            raise ValueError("fal.ai 必须提供有效的 API Key")
    return {"provider": provider, "omni_url": omni_url.strip(),
            "model": model if provider == "fal" else "",
            "api_key": api_key.strip() if provider == "fal" else ""}


def load_local_v2v(path: Path | None = None) -> dict:
    file = path or CONFIG_FILE
    if not file.exists():
        return {"provider": "mock", "omni_url": "http://127.0.0.1:8091",
                "model": FAL_EDIT_MODEL, "api_key": ""}
    data = json.loads(file.read_text(encoding="utf-8"))
    # Reject unknown config payload types/invalid values before consuming it.
    if not isinstance(data, dict):
        raise ValueError("V2V config must be an object")
    return _validate(
        str(data.get("provider", "mock")),
        str(data.get("omni_url", "http://127.0.0.1:8091")),
        str(data.get("model", FAL_EDIT_MODEL)),
        str(data.get("api_key", "")),
    )


def save_local_v2v(
    provider: str, omni_url: str, model: str,
    api_key: str, path: Path | None = None,
) -> str:
    """Atomically store with 0600 permissions; empty key preserves old fal key."""
    file = path or CONFIG_FILE
    old = load_local_v2v(file)
    secret = api_key.strip()
    if provider == "fal" and not secret and old["provider"] == "fal":
        secret = old["api_key"]
    record = _validate(provider, omni_url, model, secret)
    file.parent.mkdir(parents=True, exist_ok=True)
    # Use same-directory atomic rename and chmod as protection against crashes.
    fd, filename = tempfile.mkstemp(prefix=".v2v-", dir=str(file.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.chmod(filename, 0o600)
            json.dump(record, output, ensure_ascii=False)
        os.replace(filename, file)
        os.chmod(file, 0o600)
    finally:
        Path(filename).unlink(missing_ok=True)
    return str(file)


def ui_provider_status(path: Path | None = None) -> tuple[str, str, str, str]:
    record = load_local_v2v(path)
    return (
        record["provider"], record["omni_url"],
        record["model"] or FAL_EDIT_MODEL,
        "已设置" if record["api_key"] else "未设置",
    )
