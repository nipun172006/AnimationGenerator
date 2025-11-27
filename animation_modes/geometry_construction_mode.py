from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class GeometryConstructionConfig(AnimationConfig):
    mode: Literal["geometry_construction"] = "geometry_construction"
    description_steps: List[str]
    title: Optional[str] = None


def render_geometry_construction(cfg: GeometryConstructionConfig) -> str:
    from manim import Scene, Dot, Line, Text, VGroup, FadeIn, Transform, config

    class GeometryScene(Scene):
        def construct(self):
            # Simple triangle ABC
            a = Dot([-3, -1, 0])
            b = Dot([3, -1, 0])
            c = Dot([0, 2, 0])
            tri = VGroup(a, b, c)
            ab = Line(a.get_center(), b.get_center())
            bc = Line(b.get_center(), c.get_center())
            ca = Line(c.get_center(), a.get_center())
            lines = VGroup(ab, bc, ca)
            self.play(FadeIn(tri), FadeIn(lines))

            y_offset = 3
            for i, step in enumerate(cfg.description_steps):
                t = Text(step).scale(0.4).to_edge(1)  # UP
                t.shift([0, -0.5 * i, 0])
                self.play(FadeIn(t))
                self.wait(0.3)

            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"geometry_construction_{(cfg.title or 'construction').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = GeometryScene()
    scene.render()
    return str(file_path)
