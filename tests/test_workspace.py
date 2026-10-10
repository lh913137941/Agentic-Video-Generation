"""Creator session behavior and API routing — no GPU or web UI needed."""
import pytest
from agentic_video.workspace import (
    EDIT, STORY, VISION, initial_workspace, append_event, attach_video,
    begin_submission, choose_mode, has_active_job, plan_submission, update_task,
)


def test_one_session_retains_both_workflows():
    state = initial_workspace()
    assert state["mode"] == STORY
    source = r"C:\temp\clip.mp4"
    state = attach_video(state, source)
    assert state["mode"] == EDIT
    assert state["source_name"] is not None
    state = begin_submission(state, EDIT, {"job_id": "abc", "status": "QUEUED"},
                             summary="把视频改为黄昏")
    state = update_task(state, EDIT, {"job_id": "abc", "status": "READY"})
    state = choose_mode(state, STORY)
    state = begin_submission(
        state, STORY, {"project_id": "xyz", "shots": [{"id": 1}],
                       "status": "GENERATING"},
        summary="制作咖啡馆分镜",
    )
    assert state["edit_job"]["status"] == "READY"
    assert state["story_project"]["project_id"] == "xyz"
    assert state["active"] == STORY
    assert len(state["timeline"]) >= 4
    assert has_active_job(state, STORY)
    assert not has_active_job(state, EDIT)


def test_explicit_routing_does_not_silently_ignore_uploaded_video():
    ready = {"connected": True, "video_edit_upload": True}
    mock = {"connected": True, "video_edit_upload": False}
    assert plan_submission(EDIT, "test.mp4", "改成黄昏", 3, ready) == EDIT
    assert plan_submission(STORY, None, "分镜文案", 3, mock) == STORY
    with pytest.raises(ValueError, match="不读取上传视频"):
        plan_submission(STORY, "test.mp4", "分镜文案", 3, ready)
    with pytest.raises(ValueError, match="Mock"):
        plan_submission(EDIT, "test.mp4", "变黄昏", 3, mock)
    with pytest.raises(ValueError, match="上传 MP4"):
        plan_submission(EDIT, None, "变黄昏", 3, ready)
    with pytest.raises(ValueError, match="API"):
        plan_submission(STORY, None, "剧本", 3, {"connected": False})
    with pytest.raises(ValueError, match="请输入"):
        plan_submission(STORY, None, "   ", 3, ready)
    with pytest.raises(ValueError):
        plan_submission("not_supported", None, "hello", 3, ready)


def test_status_updates_only_append_when_changed():
    state = begin_submission(initial_workspace(), EDIT,
                             {"job_id": "abc", "status": "QUEUED"},
                             summary="修改原视频")
    state = update_task(state, EDIT, {"job_id": "abc", "status": "READY"})
    count = len(state["timeline"])
    state = update_task(state, EDIT, {"job_id": "abc", "status": "READY"})
    assert len(state["timeline"]) == count
    for _ in range(45):
        state = append_event(state, "progress")
    assert len(state["timeline"]) == 30
    with pytest.raises(ValueError):
        append_event(state, "bad", role="html")


def test_vision_mode_builds_storyboard_without_unlocking_v2v():
    cap = {"connected": True, "video_understanding": True,
           "video_edit_upload": False}
    state = attach_video(initial_workspace(), "sample.mp4")
    state = choose_mode(state, VISION)
    assert plan_submission(VISION, "sample.mp4", "", 3, cap) == VISION
    assert not cap["video_edit_upload"]
    with pytest.raises(ValueError, match="上传"):
        plan_submission(VISION, None, "analysis", 3, cap)
    with pytest.raises(ValueError, match="多模态"):
        plan_submission(VISION, "clip.mp4", "analysis", 3,
                        {"connected": True, "video_understanding": False})
    project = {"project_id": "abc", "shots": [{"id": 1}],
               "status": "DRAFT"}
    state = begin_submission(
        state, VISION,
        {"summary": "街道", "observations": ["行人"],
         "sampled_seconds": [1.0], "project": project},
        summary="理解视频",
    )
    assert state["analysis"]["summary"] == "街道"
    assert state["story_project"]["project_id"] == "abc"
    assert state["active"] == VISION
    switched = choose_mode(state, STORY)
    assert switched["analysis"] == state["analysis"]
