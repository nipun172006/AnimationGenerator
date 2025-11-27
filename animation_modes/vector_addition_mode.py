from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir
from animation_config import VectorAdditionConfig as LegacyVectorAdditionConfig, Vector2D
from main import render_vector_addition as legacy_render_vector_addition


class VectorAdditionConfig(AnimationConfig):
    mode: Literal["vector_addition"] = "vector_addition"
    vectors: List[Vector2D]
    show_resultant: bool = True
    show_tip_to_tail: bool = True
    duration_seconds: float = 10.0
    title: Optional[str] = None


def render_vector_addition(cfg: VectorAdditionConfig) -> str:
    outputs_dir = ensure_outputs_dir()
    legacy_cfg = LegacyVectorAdditionConfig(**cfg.dict())
    out_path: Path = legacy_render_vector_addition(legacy_cfg, outputs_dir)
    return str(out_path)
