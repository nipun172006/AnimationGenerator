from __future__ import annotations

from typing import Callable, Dict, Type

from .base import AnimationConfig
from .function_plot_mode import FunctionPlotConfig, render_function_plot
from .vector_addition_mode import VectorAdditionConfig, render_vector_addition
from .scatter_points_mode import ScatterPointsConfig, render_scatter_points
from .bubble_sort_visualization_mode import BubbleSortConfig, render_bubble_sort
from .pythagoras_theorem_mode import PythagorasConfig, render_pythagoras
from .text_step_derivation_mode import TextStepDerivationConfig, render_text_step_derivation
from .parametric_plot_mode import ParametricPlotConfig, render_parametric_plot
from .geometry_construction_mode import GeometryConstructionConfig, render_geometry_construction
from .matrix_visualization_mode import MatrixVisualizationConfig, render_matrix_visualization
from .number_line_interval_mode import NumberLineIntervalConfig, render_number_line_interval
from .generic_explainer_mode import GenericExplainerConfig, render_generic_explainer
from .derivative_visualization_mode import DerivativeVisualizationConfig, render_derivative_visualization
from .integral_area_visualization_mode import IntegralAreaVisualizationConfig, render_integral_area_visualization
from .limit_visualization_mode import LimitVisualizationConfig, render_limit_visualization
from .manim_code_gen_mode import ManimCodeGenConfig, render_manim_code_gen
from .matrix_mult_detailed_mode import MatrixMultDetailedConfig, render_matrix_mult_detailed

# Registry mapping
MODE_REGISTRY: Dict[str, Type[AnimationConfig]] = {
    "function_plot": FunctionPlotConfig,
    "vector_addition": VectorAdditionConfig,
    "scatter_points": ScatterPointsConfig,
    "bubble_sort_visualization": BubbleSortConfig,
    "pythagoras_theorem": PythagorasConfig,
    "text_step_derivation": TextStepDerivationConfig,
    "parametric_plot": ParametricPlotConfig,
    "geometry_construction": GeometryConstructionConfig,
    "matrix_visualization": MatrixVisualizationConfig,
    "number_line_interval": NumberLineIntervalConfig,
    "generic_explainer": GenericExplainerConfig,
    "derivative_visualization": DerivativeVisualizationConfig,
    "integral_area_visualization": IntegralAreaVisualizationConfig,
    "limit_visualization": LimitVisualizationConfig,
    "manim_code_gen": ManimCodeGenConfig,
    "matrix_mult_detailed": MatrixMultDetailedConfig,
}

RENDERER_REGISTRY: Dict[str, Callable[[AnimationConfig], str]] = {
    "function_plot": render_function_plot,
    "vector_addition": render_vector_addition,
    "scatter_points": render_scatter_points,
    "bubble_sort_visualization": render_bubble_sort,
    "pythagoras_theorem": render_pythagoras,
    "text_step_derivation": render_text_step_derivation,
    "parametric_plot": render_parametric_plot,
    "geometry_construction": render_geometry_construction,
    "matrix_visualization": render_matrix_visualization,
    "number_line_interval": render_number_line_interval,
    "generic_explainer": render_generic_explainer,
    "derivative_visualization": render_derivative_visualization,
    "integral_area_visualization": render_integral_area_visualization,
    "limit_visualization": render_limit_visualization,
    "manim_code_gen": render_manim_code_gen,
    "matrix_mult_detailed": render_matrix_mult_detailed,
}


from .sanitization import sanitize_config

def render_animation_from_mode(mode: str, data: dict) -> str:
    """Validate config and dispatch to renderer via registry."""
    if mode not in MODE_REGISTRY or mode not in RENDERER_REGISTRY:
        raise ValueError(f"Unsupported mode: {mode}")
    cfg_cls = MODE_REGISTRY[mode]
    cfg = cfg_cls(**data)
    
    # Sanitize configuration (clamp values, limit sizes, etc.)
    cfg = sanitize_config(mode, cfg)
    
    renderer = RENDERER_REGISTRY[mode]
    return renderer(cfg)
