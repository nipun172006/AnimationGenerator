from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir
from animation_config import FunctionPlotConfig as LegacyFunctionPlotConfig
from main import render_function_plot as legacy_render_function_plot


class FunctionPlotConfig(AnimationConfig):
    mode: Literal["function_plot"] = "function_plot"
    function_expression: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    duration_seconds: float
    title: str


def render_function_plot(cfg: FunctionPlotConfig) -> str:
    # Reuse existing renderer to avoid duplication
    outputs_dir = ensure_outputs_dir()
    legacy_cfg = LegacyFunctionPlotConfig.from_dict(cfg.dict())
    out_path: Path = legacy_render_function_plot(legacy_cfg, outputs_dir)
    return str(out_path)
