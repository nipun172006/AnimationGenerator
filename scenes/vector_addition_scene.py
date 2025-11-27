from __future__ import annotations

from manim import (
    Scene,
    NumberPlane,
    Arrow,
    Create,
    FadeIn,
    Text,
    BLUE,
    GREEN,
    RED,
    ORANGE,
    PURPLE,
    YELLOW,
    WHITE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)
from typing import List

from animation_config import VectorAdditionConfig, Vector2D


class VectorAdditionScene(Scene):
    """Manim scene for 2D vector addition using arrows.

    - Draws vectors from origin
    - Optionally animates tip-to-tail placement
    - Optionally draws resultant vector
    - Labels vectors and shows a title
    """

    def __init__(self, config: VectorAdditionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def construct(self):
        cfg = self.config

        # Compute dynamic extents based on vectors and mode
        def compute_extents():
            pts = [(0.0, 0.0)]
            if cfg.show_tip_to_tail:
                cx, cy = 0.0, 0.0
                for v in cfg.vectors:
                    cx += v.x
                    cy += v.y
                    pts.append((cx, cy))
            else:
                for v in cfg.vectors:
                    pts.append((v.x, v.y))

            # Include resultant tip as well
            rx = sum(v.x for v in cfg.vectors)
            ry = sum(v.y for v in cfg.vectors)
            pts.append((rx, ry))

            min_x = min(p[0] for p in pts)
            max_x = max(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)

            range_x = max(1.0, max_x - min_x)
            range_y = max(1.0, max_y - min_y)
            pad_x = max(1.0, 0.15 * range_x)
            pad_y = max(1.0, 0.15 * range_y)

            # Grid step heuristic: around ~10 grid lines
            step_x = max(0.5, round((range_x + 2 * pad_x) / 10))
            step_y = max(0.5, round((range_y + 2 * pad_y) / 10))

            return (
                min_x - pad_x,
                max_x + pad_x,
                step_x,
                min_y - pad_y,
                max_y + pad_y,
                step_y,
            )

        x_min, x_max, step_x, y_min, y_max, step_y = compute_extents()

        plane = NumberPlane(
            x_range=[x_min, x_max, step_x],
            y_range=[y_min, y_max, step_y],
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.3, "stroke_width": 1},
        )

        title = Text(cfg.title or "Vector Addition").scale(0.6)
        title.to_edge(UP, buff=0.4)

        self.play(FadeIn(plane))
        self.play(FadeIn(title, shift=DOWN))

        colors = [BLUE, GREEN, RED, ORANGE, PURPLE]

        arrows: List[Arrow] = []
        labels = []

        if cfg.show_tip_to_tail:
            # Draw a tip-to-tail chain for all vectors
            cur_x, cur_y = 0.0, 0.0
            for idx, vec in enumerate(cfg.vectors):
                color = colors[idx % len(colors)]
                start_pt = plane.c2p(cur_x, cur_y)
                end_pt = plane.c2p(cur_x + vec.x, cur_y + vec.y)
                arrow = Arrow(start=start_pt, end=end_pt, color=color)
                arrows.append(arrow)
                self.play(Create(arrow))

                label_text = f"{vec.label} ({vec.x:.2f}, {vec.y:.2f})"
                label = Text(label_text).scale(0.45)
                # Adaptive label placement to avoid edge/title overlap
                lx, ly = cur_x + vec.x, cur_y + vec.y
                direction = UP
                if ly > (y_max - (y_max - y_min) * 0.1):
                    direction = DOWN
                elif lx > (x_max - (x_max - x_min) * 0.1):
                    direction = LEFT
                elif lx < (x_min + (x_max - x_min) * 0.1):
                    direction = RIGHT
                label.next_to(end_pt, direction=direction)
                labels.append(label)
                self.play(FadeIn(label))

                cur_x += vec.x
                cur_y += vec.y
        else:
            # Draw each vector from origin independently
            for idx, vec in enumerate(cfg.vectors):
                color = colors[idx % len(colors)]
                arrow = Arrow(start=plane.c2p(0, 0), end=plane.c2p(vec.x, vec.y), color=color)
                arrows.append(arrow)
                self.play(Create(arrow))

                label_text = f"{vec.label} ({vec.x:.2f}, {vec.y:.2f})"
                label = Text(label_text).scale(0.45)
                ex, ey = vec.x, vec.y
                direction = UP
                if ey > (y_max - (y_max - y_min) * 0.1):
                    direction = DOWN
                elif ex > (x_max - (x_max - x_min) * 0.1):
                    direction = LEFT
                elif ex < (x_min + (x_max - x_min) * 0.1):
                    direction = RIGHT
                label.next_to(arrow.get_end(), direction=direction)
                labels.append(label)
                self.play(FadeIn(label))

        # Resultant vector from origin to sum of all vectors
        if cfg.show_resultant and len(cfg.vectors) >= 1:
            res_x = sum(v.x for v in cfg.vectors)
            res_y = sum(v.y for v in cfg.vectors)
            resultant = Arrow(start=plane.c2p(0, 0), end=plane.c2p(res_x, res_y), color=YELLOW)
            resultant.set_stroke(width=6)
            self.play(Create(resultant))

            res_label = Text(f"resultant ({res_x:.2f}, {res_y:.2f})").scale(0.5)
            # Avoid overlapping the title or edges
            direction = UP
            if res_y > (y_max - (y_max - y_min) * 0.1):
                direction = DOWN
            elif res_x > (x_max - (x_max - x_min) * 0.1):
                direction = LEFT
            elif res_x < (x_min + (x_max - x_min) * 0.1):
                direction = RIGHT
            res_label.next_to(resultant.get_end(), direction=direction)
            self.play(FadeIn(res_label))

        self.wait(0.6)
