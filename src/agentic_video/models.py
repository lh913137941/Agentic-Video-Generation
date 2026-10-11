"""Persistence and API schema: shot revision is independent from render revision."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ShotStatus(str, Enum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    READY = "READY"
    FAILED = "FAILED"


class ProjectStatus(str, Enum):
    DRAFT = "DRAFT"
    GENERATING = "GENERATING"
    STITCHING = "STITCHING"
    READY = "READY"
    FAILED = "FAILED"


class ShotSpec(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)
    duration: float = Field(default=3.0, ge=1, le=10)
    camera_motion: str = Field(default="static", max_length=80)


class Shot(ShotSpec):
    id: int = Field(ge=1)
    revision: int = 1
    rendered_revision: int = 0
    status: ShotStatus = ShotStatus.PENDING
    video_path: Optional[str] = None
    error: Optional[str] = None


class Project(BaseModel):
    project_id: str
    script: str
    theme_style: str
    shots: list[Shot]
    status: ProjectStatus = ProjectStatus.DRAFT
    final_video_path: Optional[str] = None
    error: Optional[str] = None


class CreateProjectRequest(BaseModel):
    script: str = Field(min_length=1, max_length=15000)
    theme_style: str = Field(default="cinematic, consistent color palette", max_length=500)
    shot_count: int = Field(default=3, ge=1, le=10)


class UpdateShotRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)
    duration: Optional[float] = Field(default=None, ge=1, le=10)
    camera_motion: Optional[str] = Field(default=None, max_length=80)
