from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class ParametricPlotConfig(AnimationConfig):
    mode: Literal["parametric_plot"] = "parametric_plot"
    x_expression: str
    y_expression: str
    t_min: float = 0.0
    t_max: float = 6.28
    duration_seconds: float = 8.0
    title: Optional[str] = None


def render_parametric_plot(cfg: ParametricPlotConfig) -> str:
    from manim import Scene, Axes, ParametricFunction, MathTex, Create, Write, config, UP, YELLOW, WHITE
    import math
    import numpy as np

    class ParametricScene(Scene):
        def construct(self):
            # Create axes
            axes = Axes(
                x_range=[-5, 5, 1],
                y_range=[-5, 5, 1],
                x_length=7,
                y_length=7,
                axis_config={"color": WHITE}
            )
            # Timing
            total_duration = max(4.0, cfg.duration_seconds)
            t_axes = total_duration * 0.2
            t_curve = total_duration * 0.7
            t_title = total_duration * 0.1

            self.play(Create(axes), run_time=t_axes)

            # Define the parametric function
            def func(t: float):
                # Safe context with standard math functions
                context = {
                    "t": t,
                    "sin": math.sin, "cos": math.cos, "tan": math.tan,
                    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
                    "pi": math.pi, "e": math.e,
                    "pow": math.pow, "abs": abs
                }
                try:
                    x = eval(cfg.x_expression, {"__builtins__": {}}, context)
                    y = eval(cfg.y_expression, {"__builtins__": {}}, context)
                except Exception:
                    x, y = 0.0, 0.0
                return axes.coords_to_point(float(x), float(y))

            # Create the curve
            curve = ParametricFunction(
                func,
                t_range=[cfg.t_min, cfg.t_max],
                color=YELLOW,
                stroke_width=4
            )
            
            self.play(Create(curve), run_time=t_curve)
            
            # Title
            if cfg.title:
                label = MathTex(cfg.title).scale(0.8).to_edge(UP)
                self.play(Write(label), run_time=t_title)
            
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'curve') if x.isalnum() or x in " -_").strip()
    file_stem = f"parametric_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    scene = ParametricScene()
    scene.render()
    
    return str(file_path)
