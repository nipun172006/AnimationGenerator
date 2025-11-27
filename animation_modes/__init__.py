from __future__ import annotations

from typing import Callable, Dict, Type

from .base import AnimationConfig
from .function_plot_mode import FunctionPlotConfig, render_function_plot
from .vector_addition_mode import VectorAdditionConfig, render_vector_addition
from .scatter_points_mode import ScatterPointsConfig, render_scatter_points
from .bubble_sort_visualization_mode import BubbleSortConfig, render_bubble_sort
from .pythagoras_theorem_mode import PythagorasConfig, render_pythagoras
from .text_step_derivation_mode import TextStepDerivationConfig, render_text_step_derivation

# Registry mapping
MODE_REGISTRY: Dict[str, Type[AnimationConfig]] = {
    "function_plot": FunctionPlotConfig,
    "vector_addition": VectorAdditionConfig,
    "scatter_points": ScatterPointsConfig,
    "bubble_sort_visualization": BubbleSortConfig,
    "pythagoras_theorem": PythagorasConfig,
    "text_step_derivation": TextStepDerivationConfig,
}

RENDERER_REGISTRY: Dict[str, Callable[[AnimationConfig], str]] = {
    "function_plot": render_function_plot,
    "vector_addition": render_vector_addition,
    "scatter_points": render_scatter_points,
    "bubble_sort_visualization": render_bubble_sort,
    "pythagoras_theorem": render_pythagoras,
    "text_step_derivation": render_text_step_derivation,
}


def render_animation_from_mode(mode: str, data: dict) -> str:
    """Validate config and dispatch to renderer via registry."""
    if mode not in MODE_REGISTRY or mode not in RENDERER_REGISTRY:
        raise ValueError(f"Unsupported mode: {mode}")
    cfg_cls = MODE_REGISTRY[mode]
    cfg = cfg_cls(**data)
    renderer = RENDERER_REGISTRY[mode]
    return renderer(cfg)
