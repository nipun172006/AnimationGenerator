from __future__ import annotations

from pathlib import Path
from typing import Protocol, Optional
from pydantic import BaseModel, Field


class AnimationConfig(BaseModel):
    mode: str = Field(..., description="Animation mode identifier")
    title: Optional[str] = Field(None, description="Optional scene title")


class AnimationRenderer(Protocol):
    def __call__(self, config: AnimationConfig) -> str:  # returns output video path
        ...


def ensure_outputs_dir() -> Path:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out
