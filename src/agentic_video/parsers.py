"""Replaceable screenplay parser; rule mode keeps the demo GPU/LLM independent."""
import json
import re
from typing import Protocol
import httpx
from pydantic import BaseModel
from .models import ShotSpec


class StoryboardParser(Protocol):
    async def parse(self, script: str, style: str, count: int) -> list[ShotSpec]: ...


class RuleStoryboardParser:
    async def parse(self, script: str, style: str, count: int) -> list[ShotSpec]:
        parts = [x.strip() for x in re.split(r"[\n。！？!?]+", script.strip()) if x.strip()]
        if not parts:
            raise ValueError("Empty script")
        if len(parts) < count:
            # For one-line scripts, offer different coverage instead of inventing events.
            views = ("establishing wide shot", "medium shot with subject action",
                     "close-up detail", "tracking shot", "closing wide shot")
            return [
                ShotSpec(prompt=f"{views[i % len(views)]}. {parts[i % len(parts)]}")
                for i in range(count)
            ]
        groups = parts[:count - 1] + ["; ".join(parts[count - 1:])]
        return [ShotSpec(prompt=part) for part in groups]


class _LLMOutput(BaseModel):
    shots: list[ShotSpec]


class OpenAIStoryboardParser:
    """An optional OpenAI-compatible chat endpoint, separate from the video model."""
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key

    async def parse(self, script: str, style: str, count: int) -> list[ShotSpec]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": (
                    f"Create exactly {count} video shots with a common style: {style}. "
                    "Return only a JSON object containing a 'shots' array. "
                    "Each shot has prompt, duration (1-10 seconds), camera_motion. "
                    "Never add unsupported facts unrelated to the script."
                )},
                {"role": "user", "content": script},
            ],
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        try:
            result = _LLMOutput.model_validate(json.loads(content))
        except (ValueError, KeyError) as exc:
            raise ValueError("LLM did not return valid storyboard JSON") from exc
        if len(result.shots) != count:
            raise ValueError(f"Expected {count} shots, received {len(result.shots)}")
        return result.shots
