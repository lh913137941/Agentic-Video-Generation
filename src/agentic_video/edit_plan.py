"""Deterministic P0 editing contract (source-clock clips, output-clock captions).

An EditPlan is auditable: model/rules may propose it but may not execute shell
commands. Human approval and the compiler are separate steps.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def valid_time(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("Transcript end must follow start")
        return self


class EditClip(BaseModel):
    source_id: Literal["source"] = "source"
    in_ms: int = Field(ge=0)
    out_ms: int = Field(gt=0)
    reason: str = Field(default="用户选择", max_length=300)

    @model_validator(mode="after")
    def valid_time(self):
        if self.out_ms - self.in_ms < 200:
            raise ValueError("Clip duration must be at least 200 ms")
        return self


class Caption(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=250)

    @model_validator(mode="after")
    def valid_time(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("Caption end must follow start")
        return self


class EditPlan(BaseModel):
    schema_version: Literal[1] = 1
    version: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=2000)
    aspect: Literal["9:16", "16:9", "1:1"] = "9:16"
    clips: list[EditClip] = Field(min_length=1, max_length=60)
    captions: list[Caption] = Field(default_factory=list, max_length=500)
    approval: Literal["pending", "approved"] = "pending"
    planner: Literal["rules", "manual", "llm"] = "manual"

    @property
    def duration_ms(self) -> int:
        return sum(c.out_ms - c.in_ms for c in self.clips)

    def validate_for_source(self, source_duration_ms: int) -> "EditPlan":
        if any(c.out_ms > source_duration_ms for c in self.clips):
            raise ValueError("Clip outside source duration")
        if self.duration_ms > 120_000:
            raise ValueError("MVP export cannot exceed 120 seconds")
        if any(c.end_ms > self.duration_ms for c in self.captions):
            raise ValueError("Caption outside edited timeline")
        return self


class PlanPatch(BaseModel):
    clips: list[EditClip] = Field(min_length=1, max_length=60)
    captions: list[Caption] = Field(default_factory=list, max_length=500)
    aspect: Literal["9:16", "16:9", "1:1"] = "9:16"


class PlanRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=2000)
    target_seconds: int = Field(default=45, ge=5, le=90)
    captions: bool = True


def validate_transcript(segments: list[TranscriptSegment], duration_ms: int) -> None:
    previous_end = 0
    for row in segments:
        if row.start_ms < previous_end or row.end_ms > duration_ms:
            raise ValueError("Transcript intervals must be ordered, nonoverlapping and inside source")
        previous_end = row.end_ms


def draft_plan(
    segments: list[TranscriptSegment], duration_ms: int,
    request: PlanRequest, version: int,
) -> EditPlan:
    """Transparent, deterministic fallback; does not claim semantic AI ranking.

    Selects sequential speech segments, excluding gaps >= 650ms. Always displays
    reasons and source timecodes for human review. No ASR -> no invented text.
    """
    validate_transcript(segments, duration_ms)
    if not segments:
        raise ValueError("No transcript; run ASR or provide reviewed segments")
    target_ms = request.target_seconds * 1000
    selected: list[TranscriptSegment] = []
    total = 0
    for row in segments:
        if total >= target_ms or len(selected) >= 60:
            break
        width = min(row.end_ms - row.start_ms, target_ms - total)
        if width < 200:
            continue
        selected.append(TranscriptSegment(
            start_ms=row.start_ms, end_ms=row.start_ms + width, text=row.text,
        ))
        total += width
    if not selected:
        raise ValueError("No usable speech in requested range")

    clips: list[EditClip] = []
    for row in selected:
        if clips and row.start_ms - clips[-1].out_ms < 650:
            clips[-1].out_ms = row.end_ms
        else:
            clips.append(EditClip(
                in_ms=row.start_ms, out_ms=row.end_ms,
                reason="ASR 语句选段；请人工确认重点内容",
            ))
    # Generated subtitle times are output-relative, not original source times.
    captions: list[Caption] = []
    if request.captions:
        cursor = 0
        for clip in clips:
            for row in selected:
                st = max(row.start_ms, clip.in_ms)
                en = min(row.end_ms, clip.out_ms)
                if en > st:
                    captions.append(Caption(
                        start_ms=cursor + st - clip.in_ms,
                        end_ms=cursor + en - clip.in_ms,
                        text=row.text[:250],
                    ))
            cursor += clip.out_ms - clip.in_ms
    aspect = "9:16" if re.search(r"竖屏|竖版|9\s*[:：/]\s*16|shorts|reels", request.instruction, re.I) else "16:9"
    plan = EditPlan(
        version=version, instruction=request.instruction, aspect=aspect,
        clips=clips, captions=captions, planner="rules",
    )
    return plan.validate_for_source(duration_ms)
