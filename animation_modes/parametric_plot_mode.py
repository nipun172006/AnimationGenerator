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
    from manim import Scene, Axes, MathTex, VMobject, config
    import math

    class ParametricScene(Scene):
        def construct(self):
            axes = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], tips=False)
            self.play(axes.create())

            def param_func(t: float):
                # VERY basic safe eval: allow t, math functions via math.
                local = {"t": t, "math": math}
                try:
                    x = eval(cfg.x_expression, {"__builtins__": {}}, local)
                    y = eval(cfg.y_expression, {"__builtins__": {}}, local)
                except Exception:
                    x, y = 0.0, 0.0
                return axes.coords_to_point(float(x), float(y))

            curve = VMobject()
            n_samples = 300
            dt = (cfg.t_max - cfg.t_min) / n_samples
            points = [param_func(cfg.t_min + i * dt) for i in range(n_samples + 1)]
            curve.set_points_smoothly(points)
            self.play(curve.animate.set_color("YELLOW"))
            self.wait(1)

            label = MathTex(cfg.title or "Parametric Curve").scale(0.6).to_edge(1)  # UP
            self.play(label.animate.set_color("WHITE"))
            self.wait(0.5)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"parametric_plot_{(cfg.title or 'curve').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = ParametricScene()
    scene.render()
    return str(file_path)
