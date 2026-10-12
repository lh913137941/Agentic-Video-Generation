"""P0 product gates: no GPU or commercial API required."""
import json
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from agentic_video.edit_plan import (
    Caption, EditClip, EditPlan, PlanRequest, TranscriptSegment, draft_plan,
)
from agentic_video.api import create_app, build_service
from agentic_video.config import Settings


def test_source_clock_and_output_clock_and_fail_closed():
    segments = [
        TranscriptSegment(start_ms=1000, end_ms=2200, text="第一句话"),
        TranscriptSegment(start_ms=4200, end_ms=5500, text="第二句话"),
    ]
    p = draft_plan(segments, 6000, PlanRequest(
        instruction="剪成竖屏并去停顿", target_seconds=5,
    ), 1)
    assert p.aspect == "9:16"
    assert len(p.clips) == 2
    assert p.clips[0].in_ms == 1000
    assert p.clips[1].in_ms == 4200
    assert p.captions[1].start_ms == 1200
    assert p.approval == "pending"
    assert p.planner == "rules"
    with pytest.raises(ValueError, match="outside source"):
        EditPlan(version=1, instruction="manual",
                 clips=[EditClip(in_ms=5, out_ms=9000)]).validate_for_source(6000)
    with pytest.raises(ValueError, match="Caption outside"):
        EditPlan(version=1, instruction="manual",
                 clips=[EditClip(in_ms=0, out_ms=1000)],
                 captions=[Caption(start_ms=0, end_ms=1500, text="bad")]
                 ).validate_for_source(6000)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="requires actual FFmpeg")
def test_upload_review_caption_render_and_retry_are_real(tmp_path, monkeypatch):
    monkeypatch.setenv("AVG_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=256x144:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "3", "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p", "-threads", "2", str(src),
    ], check=True, timeout=50)
    service = build_service(Settings(data_dir=tmp_path / "data"))
    app = create_app(service)
    app.state.p0_editing.render_edge = 144  # keep CI CPU cost tiny
    with TestClient(app) as client:
        with src.open("rb") as stream:
            created = client.post("/api/mvp/projects", files={
                "file": ("source.mp4", stream, "video/mp4")
            })
        assert created.status_code == 201, created.text
        project_id = created.json()["id"]
        assert created.json()["source_has_audio"] is True
        assert client.get("/api/mvp/projects/" + project_id + "/source").status_code == 200

        r = client.put(f"/api/mvp/projects/{project_id}/transcript", json={
            "segments": [
                {"start_ms": 200, "end_ms": 1200, "text": "镜头一"},
                {"start_ms": 1900, "end_ms": 2700, "text": "镜头二"},
            ]
        })
        assert r.status_code == 200, r.text
        p = client.post(f"/api/mvp/projects/{project_id}/plans", json={
            "instruction": "去停顿，转为9:16竖屏，自动字幕",
            "target_seconds": 5,
        })
        assert p.status_code == 200, p.text
        plan = p.json()
        assert plan["approval"] == "pending" and len(plan["clips"]) == 2
        assert client.post(f"/api/mvp/projects/{project_id}/renders?version=1").status_code == 409
        # Manual correction creates revision 2 and resets human approval.
        update = client.patch(f"/api/mvp/projects/{project_id}/plans/1", json={
            "clips": plan["clips"], "captions": plan["captions"], "aspect": "9:16",
        })
        assert update.status_code == 200, update.text
        assert update.json()["version"] == 2
        assert client.post(f"/api/mvp/projects/{project_id}/plans/1/approve").status_code == 409
        assert client.post(f"/api/mvp/projects/{project_id}/plans/2/approve").status_code == 200
        queued = client.post(f"/api/mvp/projects/{project_id}/renders?version=2")
        assert queued.status_code == 202, queued.text
        job_id = queued.json()["id"]
        repeated = client.post(f"/api/mvp/projects/{project_id}/renders?version=2")
        assert repeated.json()["id"] == job_id
        state = None
        for _ in range(160):
            state = client.get(f"/api/mvp/projects/{project_id}/renders/{job_id}").json()
            if state["status"] in ("READY", "FAILED"):
                break
            time.sleep(.15)
        assert state["status"] == "READY", state
        assert state["metrics"]["width"] == 144
        assert state["metrics"]["height"] == 256
        assert state["metrics"]["has_audio"] is True
        assert state["metrics"]["captions"] == 2
        assert client.get(f"/api/mvp/projects/{project_id}/renders/{job_id}/file").content[4:8] == b"ftyp"
        assert client.post(f"/api/mvp/projects/{project_id}/renders?version=2").json()["id"] == job_id
    # Completed results and edit plans persist across restarts.
    app2 = create_app(build_service(Settings(data_dir=tmp_path / "data")))
    with TestClient(app2) as client:
        assert client.get(f"/api/mvp/projects/{project_id}/plans/2").json()["approval"] == "approved"
        assert client.get(f"/api/mvp/projects/{project_id}/renders/{job_id}").json()["status"] == "READY"


def test_malformed_timestamps_rejected():
    with pytest.raises(ValueError):
        draft_plan([TranscriptSegment(start_ms=500, end_ms=1800, text="late"),
                    TranscriptSegment(start_ms=1200, end_ms=1900, text="overlap")],
                   2500, PlanRequest(instruction="摘要"), version=1)

def test_llm_plan_is_grounded_and_unapproved():
    from agentic_video.edit_agent import plan_from_ids
    rows = [
        TranscriptSegment(start_ms=500, end_ms=1500, text="开头介绍"),
        TranscriptSegment(start_ms=5000, end_ms=6500, text="关键观点"),
        TranscriptSegment(start_ms=7300, end_ms=8500, text="结束语"),
    ]
    request = PlanRequest(instruction="保留关键观点，改成9:16竖屏", target_seconds=5)
    result = plan_from_ids(rows, [1], request, 2, 9000)
    assert result.planner == "llm"
    assert result.approval == "pending"
    assert result.clips[0].in_ms == 5000
    assert result.captions[0].start_ms == 0
    assert result.aspect == "9:16"
    with pytest.raises(ValueError, match="duplicate"):
        plan_from_ids(rows, [1, 1], request, 2, 9000)
    with pytest.raises(ValueError, match="nonexistent"):
        plan_from_ids(rows, [3], request, 2, 9000)
    with pytest.raises(ValueError, match="unordered"):
        plan_from_ids(rows, [2, 1], request, 2, 9000)

@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="requires actual FFmpeg")
def test_no_audio_source_renders_with_explicit_silent_track(tmp_path):
    from agentic_video.edit_executor import probe_media, render_edit_plan
    source = tmp_path / "mute.mp4"
    target = tmp_path / "vertical.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=12",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-threads", "2", str(source)
    ], check=True, timeout=50)
    original = source.read_bytes()
    p = EditPlan(version=1, instruction="mute preview", approval="approved",
                 aspect="9:16",
                 clips=[EditClip(in_ms=200, out_ms=1400)])
    report = render_edit_plan(source, p, target, short_edge=144)
    assert report["source_has_audio"] is False
    assert report["has_audio"] is True
    assert report["width"] == 144 and report["height"] == 256
    assert probe_media(target)[1] is True
    assert source.read_bytes() == original
