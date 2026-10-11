"""Single-page creator session state and explicit API workflow routing.

This is presentation/session state only. The REST backend still keeps
video edits and storyboard projects in their existing stores.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

EDIT = "video_edit"
STORY = "storyboard"
VISION = "video_understanding"
MODES = (EDIT, STORY, VISION)
ACTIVE_STATUSES = ("QUEUED", "GENERATING", "STITCHING")


def initial_workspace() -> dict[str, Any]:
    return {
        "mode": STORY,
        "active": STORY,
        "edit_job": None,
        "story_project": None,
        "analysis": None,
        "annotation": None,
        "source_name": None,
        "timeline": [],
    }


def append_event(
    state: dict[str, Any], message: str, *, role: str = "system"
) -> dict[str, Any]:
    next_state = deepcopy(state or initial_workspace())
    if role not in ("user", "system"):
        raise ValueError("Invalid timeline role")
    next_state.setdefault("timeline", []).append(
        {"role": role, "message": str(message)[:400]}
    )
    next_state["timeline"] = next_state["timeline"][-30:]
    return next_state


def choose_mode(state: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("Unsupported workflow")
    next_state = deepcopy(state or initial_workspace())
    next_state["mode"] = mode
    # Selecting a skill also selects its own result in the same workspace.
    next_state["active"] = mode
    return next_state


def attach_video(state: dict[str, Any], file_path: str | None) -> dict[str, Any]:
    next_state = deepcopy(state or initial_workspace())
    next_name = Path(file_path).name if file_path else None
    previous = next_state.get("source_name")
    if next_name != previous:
        next_state["source_name"] = next_name
        next_state["annotation"] = None
        if next_name:
            next_state = choose_mode(next_state, EDIT)
            next_state = append_event(next_state, "已添加视频素材，可使用「视频编辑」")
        elif previous:
            next_state = append_event(next_state, "已移除上传的视频素材")
    return next_state


def plan_submission(
    mode: str, source: str | None, text: str, shot_count: int,
    capabilities: dict[str, Any],
) -> str:
    """Return the existing API workflow; never silently reinterpret a file."""
    if not capabilities.get("connected"):
        raise ValueError("API 尚未连接，请先启动后端并重新检测")
    if mode == VISION:
        if not source:
            raise ValueError("视频理解需要先上传 MP4")
        if not capabilities.get("video_understanding"):
            raise ValueError("尚未配置 OpenAI 兼容多模态视觉模型")
        if not 1 <= int(shot_count) <= 10:
            raise ValueError("分镜数量需要在 1～10 之间")
        return VISION
    if not text or not text.strip():
        raise ValueError("请输入视频编辑指令或剧本内容")
    if mode == EDIT:
        if not source:
            raise ValueError("视频编辑需要先上传 MP4 原视频")
        if not capabilities.get("video_edit_upload"):
            raise ValueError(
                "当前为 Mock 模式，无法编辑上传的视频；请选择「创作分镜」"
                "或部署支持 V2V 的推理模型"
            )
        return EDIT
    if mode == STORY:
        if source:
            raise ValueError(
                "「创作分镜」目前不读取上传视频。请移除素材，"
                "或切换「视频编辑」，避免误认为视频被用作生成条件"
            )
        if not 1 <= int(shot_count) <= 10:
            raise ValueError("分镜数量需要在 1～10 之间")
        return STORY
    raise ValueError("请选择可用的创作技能")


def begin_submission(
    state: dict[str, Any], mode: str, payload: dict[str, Any],
    *, summary: str,
) -> dict[str, Any]:
    next_state = choose_mode(state, mode)
    next_state = append_event(next_state, summary, role="user")
    if mode == VISION:
        next_state["analysis"] = payload
        next_state["story_project"] = payload["project"]
        next_state = append_event(
            next_state, "关键帧理解完成，已保存结构化分镜草稿；可检查后生成粗剪"
        )
    elif mode == EDIT:
        next_state["edit_job"] = payload
        next_state = append_event(next_state, "视频编辑任务已提交，等待推理服务处理")
    else:
        next_state["story_project"] = payload
        next_state = append_event(
            next_state,
            f"已创建 {len(payload.get('shots', []))} 个分镜，开始逐镜头生成",
        )
    return next_state


def update_task(
    state: dict[str, Any], mode: str, payload: dict[str, Any],
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("Unsupported workflow")
    next_state = deepcopy(state or initial_workspace())
    key = "edit_job" if mode == EDIT else "story_project"
    before = next_state.get(key) or {}
    if payload == before:
        return next_state
    next_state[key] = payload
    old_status = before.get("status")
    new_status = payload.get("status")
    if new_status and new_status != old_status:
        labels = {
            "QUEUED": "等待生成", "GENERATING": "正在生成",
            "STITCHING": "正在合成粗剪", "READY": "生成完成，可预览导出",
            "FAILED": "任务失败，请查看错误信息",
            "DRAFT": "分镜待处理",
        }
        label = "视频编辑" if mode == EDIT else "分镜粗剪"
        next_state = append_event(
            next_state, f"{label} · {labels.get(new_status, new_status)}"
        )
    return next_state


def has_active_job(state: dict[str, Any], mode: str) -> bool:
    key = "edit_job" if mode == EDIT else "story_project"
    task = (state or {}).get(key) or {}
    return task.get("status") in ACTIVE_STATUSES


def record_annotation(
    state: dict[str, Any], preview: str, manifest: str,
    instruction: str, frame_second: float,
) -> dict[str, Any]:
    """Store the marked PNG and metadata within the current UI session."""
    if not preview or not manifest or not instruction.strip():
        raise ValueError("Incomplete saved annotation")
    next_state = append_event(
        state, f"已保存 {frame_second:.2f}s 画面标记，修改要求：{instruction[:140]}"
    )
    next_state["annotation"] = {
        "preview": preview, "manifest": manifest,
        "instruction": instruction, "frame_second": float(frame_second),
    }
    return next_state
