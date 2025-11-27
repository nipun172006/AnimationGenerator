from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class NumberLineIntervalConfig(AnimationConfig):
    mode: Literal["number_line_interval"] = "number_line_interval"
    interval_type: Literal["<", ">", "<=", ">="]
    value: float
    title: Optional[str] = None


def render_number_line_interval(cfg: NumberLineIntervalConfig) -> str:
    from manim import Scene, NumberLine, Dot, Arrow, Text, FadeIn, config

    class IntervalScene(Scene):
        def construct(self):
            number_line = NumberLine(x_range=[cfg.value - 5, cfg.value + 5, 1])
            self.play(FadeIn(number_line))
            dot = Dot(point=number_line.n2p(cfg.value))
            filled = cfg.interval_type in ["<=", ">="]
            if not filled:
                # Represent open circle by scaling and setting stroke
                dot.set_color("WHITE").set_stroke("WHITE", width=2).set_fill(opacity=0)
            self.play(FadeIn(dot))

            direction = 1 if cfg.interval_type in [">", ">="] else -1
            arrow = Arrow(start=dot.get_center(), end=dot.get_center() + [direction * 3.5, 0, 0])
            self.play(FadeIn(arrow))
            label = Text(cfg.title or f"x {cfg.interval_type} {cfg.value}").scale(0.5).to_edge(1)  # UP
            self.play(FadeIn(label))
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"number_line_interval_{(cfg.title or 'interval').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = IntervalScene()
    scene.render()
    return str(file_path)
