"""Constrained optional LLM planner: LLM selects *existing ASR segment IDs* only.

The model cannot create media paths, executable code, timecodes or captions.
The server compiles grounded segments into an EditPlan, then humans approve.
"""
from __future__ import annotations

import json
import os

import httpx

from .edit_plan import Caption, EditClip, EditPlan, PlanRequest, TranscriptSegment


def plan_from_ids(
    segments: list[TranscriptSegment],
    ids: list[int],
    request: PlanRequest,
    version: int,
    source_duration_ms: int,
) -> EditPlan:
    if not ids or len(ids) > 60 or any(type(i) is not int for i in ids):
        raise ValueError("Model did not return valid segment IDs")
    if ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= len(segments):
        raise ValueError("Model returned duplicate, unordered or nonexistent segments")
    clips = []
    captions = []
    cursor = 0
    target = request.target_seconds * 1000
    for i in ids:
        seg = segments[i]
        length = min(seg.end_ms - seg.start_ms, target - cursor)
        if length < 200:
            break
        clips.append(EditClip(
            in_ms=seg.start_ms, out_ms=seg.start_ms + length,
            reason=f"LLM 从 ASR 候选语句 {i} 选段；等待人工审阅",
        ))
        if request.captions:
            captions.append(Caption(
                start_ms=cursor, end_ms=cursor + length, text=seg.text[:250],
            ))
        cursor += length
    if not clips:
        raise ValueError("Model selected no usable speech")
    instruction = request.instruction.lower()
    vertical = any(x in instruction for x in ("9:16", "9：16", "竖屏", "竖版", "reels", "shorts"))
    return EditPlan(
        version=version, instruction=request.instruction,
        aspect="9:16" if vertical else "16:9",
        planner="llm", clips=clips, captions=captions, approval="pending",
    ).validate_for_source(source_duration_ms)


async def select_segment_ids(
    segments: list[TranscriptSegment], request: PlanRequest,
    *, base_url: str, model: str, api_key: str = "",
) -> list[int]:
    if not model.strip():
        raise RuntimeError("Set AVG_P0_LLM_MODEL to enable the LLM planner")
    url = httpx.URL(base_url)
    if (not url.host or url.userinfo or url.query or url.fragment or
        (url.scheme != "https" and not (
            url.scheme == "http" and url.host in {"localhost", "127.0.0.1", "::1"}
        ))):
        raise ValueError("LLM endpoint must be HTTPS or localhost")
    if len(segments) > 300:
        raise ValueError("LLM MVP supports up to 300 transcript segments; use manual review for longer transcripts")
    route = base_url.rstrip("/")
    route += "/chat/completions" if route.endswith("/v1") else "/v1/chat/completions"
    rows = [
        {"id": i, "start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text}
        for i, s in enumerate(segments)
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": (
                "You are a transcript-only video editing planner. Select existing segment IDs "
                "for a coherent short clip. Return ONLY a JSON object with a segment_ids "
                "integer array sorted in ascending order, no duplicates, max 60 IDs. "
                "Choose segments based on topic relevance and narrative continuity, "
                "target the requested duration, omit filler/pauses. Do not invent timestamps, "
                "files, unsupported content, or claim to have analyzed video images. "
                "The selected segments will be shown for human approval."
            )},
            {"role": "user", "content": json.dumps({
                "instruction": request.instruction,
                "target_duration_ms": request.target_seconds * 1000,
                "transcript_segments": rows,
            }, ensure_ascii=False)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
            response = await client.post(route, json=payload, headers=headers)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("Model returned no text")
        result = json.loads(raw)
        ids = result.get("segment_ids")
        if not isinstance(ids, list):
            raise ValueError("Model returned no segment_ids array")
        return ids
    except httpx.HTTPError as exc:
        raise RuntimeError("LLM planning service failed or timed out") from exc
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Model did not return valid structured JSON") from exc
