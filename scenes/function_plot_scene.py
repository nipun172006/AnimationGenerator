from __future__ import annotations

from manim import Scene, Axes, Create, FadeIn, Text, BLUE, WHITE, UP, DOWN
from animation_config import FunctionPlotConfig


class FunctionPlotScene(Scene):
    """Manim scene that plots a function over given x/y ranges with a title.

    Keep the animation simple and clean for STEP 1.
    """

    def __init__(self, config: FunctionPlotConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def construct(self):
        cfg = self.config

        # Build axes; choose reasonable visual dimensions.
        # Provide explicit step to avoid dtype inference issues inside Manim
        x_step = max(0.5, (cfg.x_max - cfg.x_min) / 8.0)
        y_step = max(0.5, (cfg.y_max - cfg.y_min) / 6.0)
        axes = Axes(
            x_range=[cfg.x_min, cfg.x_max, x_step],
            y_range=[cfg.y_min, cfg.y_max, y_step],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False,
        )

        # Title at top
        title = Text(cfg.title).scale(0.6)
        title.to_edge(UP)

        # Plot the function
        graph = axes.plot(lambda x: cfg.function_callable(x), color=BLUE)

        # Animate: fade in title, draw axes, then draw graph over duration
        self.play(FadeIn(title, shift=DOWN))
        self.play(Create(axes), run_time=1.2)

        # Use most of duration for drawing the graph, keep a small tail to settle.
        draw_time = max(0.8, cfg.duration_seconds - 1.8)
        self.play(Create(graph), run_time=draw_time)
        self.wait(0.6)
