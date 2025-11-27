from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class MatrixVisualizationConfig(AnimationConfig):
    mode: Literal["matrix_visualization"] = "matrix_visualization"
    matrix: List[List[float]]
    highlight: Literal["row", "column"] = "row"
    index: int = 0
    title: Optional[str] = None


def render_matrix_visualization(cfg: MatrixVisualizationConfig) -> str:
    from manim import Scene, VGroup, Rectangle, Text, FadeIn, config

    class MatrixScene(Scene):
        def construct(self):
            rows = len(cfg.matrix)
            cols = len(cfg.matrix[0]) if rows else 0
            cell_group = VGroup()
            for r in range(rows):
                for c in range(cols):
                    rect = Rectangle(width=1.0, height=0.6)
                    rect.shift([c * 1.1 - cols * 0.55, rows * 0.3 - r * 0.7, 0])
                    label = Text(str(cfg.matrix[r][c])).scale(0.4).move_to(rect.get_center())
                    cell_group.add(rect, label)
            self.play(FadeIn(cell_group))

            # Highlight row or column
            highlights = VGroup()
            if cfg.highlight == "row" and 0 <= cfg.index < rows:
                for i in range(cols):
                    # Rectangle indices: each cell has two entries (rect,label)
                    rect_idx = (cfg.index * cols + i) * 2
                    rect = cell_group[rect_idx]
                    rect.set_color("YELLOW")
            elif cfg.highlight == "column" and 0 <= cfg.index < cols:
                for i in range(rows):
                    rect_idx = (i * cols + cfg.index) * 2
                    rect = cell_group[rect_idx]
                    rect.set_color("YELLOW")
            self.wait(1)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"matrix_visualization_{(cfg.title or 'matrix').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = MatrixScene()
    scene.render()
    return str(file_path)
