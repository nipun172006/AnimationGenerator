from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, BaseModel
from .base import AnimationConfig, ensure_outputs_dir

# Minimal scatter points config and renderer

class Point2D(BaseModel):
    label: str
    x: float
    y: float


class ScatterPointsConfig(AnimationConfig):
    mode: Literal["scatter_points"] = "scatter_points"
    points: List[Point2D]
    title: Optional[str] = None


def render_scatter_points(cfg: ScatterPointsConfig) -> str:
    # Simple Manim scene inline to avoid heavy refactor here
    from manim import Scene, NumberPlane, Dot, Text, VGroup, FadeIn

    class ScatterScene(Scene):
        def construct(self):
            plane = NumberPlane()
            self.play(FadeIn(plane))
            group = VGroup()
            for p in cfg.points:
                d = Dot(plane.c2p(p.x, p.y))
                lbl = Text(p.label).scale(0.5).next_to(d, direction=1)  # RIGHT
                group.add(d, lbl)
            self.play(FadeIn(group))
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"scatter_points_{(cfg.title or 'points').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename

    # Render
    from manim import config
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = ScatterScene()
    scene.render()

    return str(file_path)
