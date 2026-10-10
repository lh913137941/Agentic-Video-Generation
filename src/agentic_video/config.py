"""Configuration from environment; no process-global mutable state."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"
    parser: str = "rule"
    data_dir: Path = Path("./data")
    omni_url: str = "http://127.0.0.1:8091"
    llm_url: str = "http://127.0.0.1:8001"
    llm_model: str = ""
    llm_api_key: str = ""
    vision_url: str = "https://api.openai.com"
    vision_model: str = ""
    vision_api_key: str = ""
    vision_frames: int = 4
    vision_timeout_seconds: int = 120
    max_shots: int = 5
    generation_concurrency: int = 1
    width: int = 832
    height: int = 480
    model_fps: int = 16
    model_steps: int = 40
    model_frame_stride: int = 4  # 4k+1 frames for Wan; 0 disables alignment.
    output_width: int = 1280
    output_height: int = 720
    output_fps: int = 24
    job_timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.backend not in {"mock", "omni"}:
            raise ValueError("AVG_BACKEND must be mock or omni")
        if self.parser not in {"rule", "llm"}:
            raise ValueError("AVG_PARSER must be rule or llm")
        if self.parser == "llm" and not self.llm_model:
            raise ValueError("AVG_LLM_MODEL is required for llm parser")
        positive = (
            self.max_shots, self.generation_concurrency, self.width, self.height,
            self.model_fps, self.model_steps, self.output_width,
            self.output_height, self.output_fps, self.job_timeout_seconds,
        )
        if not 1 <= self.vision_frames <= 8:
            raise ValueError("AVG_VISION_FRAMES must be 1..8")
        if not 10 <= self.vision_timeout_seconds <= 600:
            raise ValueError("AVG_VISION_TIMEOUT_SECONDS must be 10..600")
        if self.model_frame_stride < 0:
            raise ValueError("model_frame_stride must be >= 0")
        if any(x < 1 for x in positive) or self.max_shots > 10:
            raise ValueError("Invalid settings: positive numeric values required; max_shots <= 10")

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            backend=env.get("AVG_BACKEND", "mock").lower(),
            parser=env.get("AVG_PARSER", "rule").lower(),
            data_dir=Path(env.get("AVG_DATA_DIR", "./data")),
            omni_url=env.get("AVG_OMNI_URL", "http://127.0.0.1:8091"),
            llm_url=env.get("AVG_LLM_URL", "http://127.0.0.1:8001"),
            llm_model=env.get("AVG_LLM_MODEL", ""),
            llm_api_key=env.get("AVG_LLM_API_KEY", ""),
            vision_url=env.get("AVG_VISION_URL", "https://api.openai.com"),
            vision_model=env.get("AVG_VISION_MODEL", ""),
            vision_api_key=env.get("AVG_VISION_API_KEY", ""),
            vision_frames=int(env.get("AVG_VISION_FRAMES", "4")),
            vision_timeout_seconds=int(env.get("AVG_VISION_TIMEOUT_SECONDS", "120")),
            max_shots=int(env.get("AVG_MAX_SHOTS", "5")),
            generation_concurrency=int(env.get("AVG_GENERATION_CONCURRENCY", "1")),
            width=int(env.get("AVG_WIDTH", "832")),
            height=int(env.get("AVG_HEIGHT", "480")),
            model_fps=int(env.get("AVG_MODEL_FPS", "16")),
            model_steps=int(env.get("AVG_MODEL_STEPS", "40")),
            model_frame_stride=int(env.get("AVG_MODEL_FRAME_STRIDE", "4")),
            output_width=int(env.get("AVG_OUTPUT_WIDTH", "1280")),
            output_height=int(env.get("AVG_OUTPUT_HEIGHT", "720")),
            output_fps=int(env.get("AVG_OUTPUT_FPS", "24")),
            job_timeout_seconds=int(env.get("AVG_JOB_TIMEOUT_SECONDS", "1800")),
        )
