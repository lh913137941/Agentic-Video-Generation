from fastapi.testclient import TestClient

from agentic_video.api import create_app
from agentic_video.parsers import RuleStoryboardParser
from agentic_video.service import VideoService
from agentic_video.store import FileProjectStore


class Backend:
    async def generate(self, prompt, destination, duration):
        destination.write_bytes(b"clip")


class Composer:
    def compose(self, clips, output):
        output.write_bytes(b"video")


def test_rest_contract(tmp_path):
    service = VideoService(
        FileProjectStore(tmp_path), RuleStoryboardParser(), Backend(), Composer()
    )
    with TestClient(create_app(service)) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        response = client.post("/api/projects", json={
            "script": "First scene. Second scene.", "shot_count": 2
        })
        assert response.status_code == 201
        pid = response.json()["project_id"]
        assert client.get("/api/projects/bad-id").status_code == 404
        assert client.get(f"/api/projects/{pid}/video").status_code == 409
        assert client.patch(
            f"/api/projects/{pid}/shots/99", json={"prompt": "hello"}
        ).status_code == 404
        assert client.post(f"/api/projects/{pid}/generate").status_code == 202
        import time
        for _ in range(100):
            p = client.get(f"/api/projects/{pid}").json()
            if p["status"] == "READY":
                break
            time.sleep(0.02)
        assert p["status"] == "READY", p
        assert client.get(f"/api/projects/{pid}/video").content == b"video"
        assert client.get(
            f"/api/projects/{pid}/shots/1/video"
        ).content == b"clip"
