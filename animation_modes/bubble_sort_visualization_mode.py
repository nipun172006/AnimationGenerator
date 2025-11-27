from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class BubbleSortConfig(AnimationConfig):
    mode: Literal["bubble_sort_visualization"] = "bubble_sort_visualization"
    array: List[int]
    duration_seconds: float = 10.0
    title: Optional[str] = None


def render_bubble_sort(cfg: BubbleSortConfig) -> str:
    # Minimal bubble sort animation using rectangles scaled by value
    from manim import Scene, VGroup, Rectangle, Text, DOWN, RIGHT, FadeIn, Transform

    class BubbleSortScene(Scene):
        def construct(self):
            values = cfg.array[:]
            bars = VGroup()
            for i, v in enumerate(values):
                bar = Rectangle(width=0.6, height=max(0.2, 0.2 + 0.2 * v))
                bar.shift(RIGHT * (i * 0.8))
                label = Text(str(v)).scale(0.4).next_to(bar, DOWN)
                bars.add(bar, label)
            bars.center()
            self.play(FadeIn(bars))

            n = len(values)
            for i in range(n):
                for j in range(0, n - i - 1):
                    # If order wrong, swap simple positions
                    if values[j] > values[j + 1]:
                        idx_a = j * 2
                        idx_b = (j + 1) * 2
                        a_bar, a_lbl = bars[idx_a], bars[idx_a + 1]
                        b_bar, b_lbl = bars[idx_b], bars[idx_b + 1]
                        self.play(
                            Transform(a_bar, a_bar.copy().shift(RIGHT * 0.8)),
                            Transform(a_lbl, a_lbl.copy().shift(RIGHT * 0.8)),
                            Transform(b_bar, b_bar.copy().shift(RIGHT * -0.8)),
                            Transform(b_lbl, b_lbl.copy().shift(RIGHT * -0.8)),
                        )
                        # Logical swap
                        values[j], values[j + 1] = values[j + 1], values[j]
                        # Swap in group (maintain order)
                        bars[idx_a], bars[idx_b] = b_bar, a_bar
                        bars[idx_a + 1], bars[idx_b + 1] = b_lbl, a_lbl
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"bubble_sort_{(cfg.title or 'array').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename

    from manim import config
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = BubbleSortScene()
    scene.render()

    return str(file_path)
