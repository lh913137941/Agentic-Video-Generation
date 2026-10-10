"""UI wiring smoke test: no service or GPU is required to build the Gradio Blocks."""
import pytest


def test_gradio_components_construct():
    pytest.importorskip("gradio")
    from agentic_video.ui import build_ui
    demo = build_ui("http://127.0.0.1:8000")
    try:
        assert demo is not None
    finally:
        demo.close()
