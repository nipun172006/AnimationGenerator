from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class PythagorasConfig(AnimationConfig):
    mode: Literal["pythagoras_theorem"] = "pythagoras_theorem"
    a: float
    b: float
    title: Optional[str] = None


def render_pythagoras(cfg: PythagorasConfig) -> str:
    # Minimal visual: right triangle with squares on legs and hypotenuse
    from manim import Scene, Square, Polygon, VGroup, Text, FadeIn, Create

    class PythagorasScene(Scene):
        def construct(self):
            a = cfg.a
            b = cfg.b
            # Triangle points
            A = (-a / 2, -b / 2, 0)
            B = (a / 2, -b / 2, 0)
            C = (-a / 2, b / 2, 0)
            tri = Polygon(A, B, C)
            self.play(Create(tri))
            # Squares (conceptual, scaled)
            sq_a = Square(side_length=abs(a)).next_to(tri, direction=1)  # RIGHT
            sq_b = Square(side_length=abs(b)).next_to(tri, direction=2)  # UP
            c = (a ** 2 + b ** 2) ** 0.5
            sq_c = Square(side_length=c).shift((0, 0, 0))
            labels = VGroup(
                Text(f"a={a}").scale(0.5),
                Text(f"b={b}").scale(0.5),
                Text(f"c={c:.2f}").scale(0.5),
            )
            self.play(FadeIn(sq_a), FadeIn(sq_b), FadeIn(sq_c), FadeIn(labels))
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"pythagoras_{(cfg.title or 'triangle').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename

    from manim import config
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = PythagorasScene()
    scene.render()

    return str(file_path)
