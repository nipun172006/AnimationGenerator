from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class NumberLineIntervalConfig(AnimationConfig):
    mode: Literal["number_line_interval"] = "number_line_interval"
    # Old fields
    interval_type: Optional[str] = None
    value: Optional[float] = None
    
    # New fields from user prompt
    left_value: Optional[float] = None
    right_value: Optional[float] = None
    include_left: bool = False
    include_right: bool = False
    
    duration_seconds: float = 8.0
    title: Optional[str] = None


def render_number_line_interval(cfg: NumberLineIntervalConfig) -> str:
    from manim import Scene, NumberLine, Dot, Arrow, Text, FadeIn, config, Line, LEFT, RIGHT, UP, DOWN, ORIGIN
    
    class IntervalScene(Scene):
        def construct(self):
            # Determine range
            val_min = cfg.left_value if cfg.left_value is not None else (cfg.value if cfg.value is not None else -5)
            val_max = cfg.right_value if cfg.right_value is not None else (cfg.value if cfg.value is not None else 5)
            
            # If single value (old style or ray)
            if cfg.left_value is None and cfg.right_value is None and cfg.value is not None:
                # Fallback to old behavior
                center = cfg.value
                span = 5
            else:
                center = (val_min + val_max) / 2
                span = max(5, (val_max - val_min) / 2 + 2)

            number_line = NumberLine(x_range=[center - span, center + span, 1], length=10, include_numbers=False)
            self.play(FadeIn(number_line))
            
            # Manually add numbers using Text (avoids LaTeX dependency)
            min_val = int(center - span)
            max_val = int(center + span)
            
            for x in range(min_val, max_val + 1):
                label = Text(str(x), font_size=20).next_to(number_line.n2p(x), DOWN, buff=0.2)
                self.add(label)
            
            # Title
            if cfg.title:
                t = Text(cfg.title).to_edge(UP)
                self.play(FadeIn(t))

            # Render Interval
            # Case 1: Finite Interval (left and right)
            if cfg.left_value is not None and cfg.right_value is not None:
                p1 = number_line.n2p(cfg.left_value)
                p2 = number_line.n2p(cfg.right_value)
                
                # Segment
                segment = Line(p1, p2, color="YELLOW", stroke_width=6)
                self.play(FadeIn(segment))
                
                # Endpoints
                dot1 = Dot(p1, color="YELLOW")
                if not cfg.include_left:
                    dot1.set_fill(color="BLACK").set_stroke(color="YELLOW", width=2)
                
                dot2 = Dot(p2, color="YELLOW")
                if not cfg.include_right:
                    dot2.set_fill(color="BLACK").set_stroke(color="YELLOW", width=2)
                    
                self.play(FadeIn(dot1), FadeIn(dot2))
                
            # Case 2: Ray (only left or only right) or single value
            elif cfg.value is not None:
                 # Old logic for simple inequalities x > 5
                 p = number_line.n2p(cfg.value)
                 dot = Dot(p, color="YELLOW")
                 
                 # Determine direction from interval_type
                 direction = 1 # Default right
                 filled = True
                 if cfg.interval_type in ["<", "<="]:
                     direction = -1
                 if cfg.interval_type in ["<", ">"]:
                     filled = False
                 
                 if not filled:
                     dot.set_fill(color="BLACK").set_stroke(color="YELLOW", width=2)
                 
                 arrow = Arrow(start=p, end=p + [direction * 4, 0, 0], color="YELLOW", buff=0)
                 self.play(FadeIn(dot), FadeIn(arrow))

            self.wait(2)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(x for x in (cfg.title or 'interval') if x.isalnum() or x in " -_").strip()
    file_stem = f"number_line_interval_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    scene = IntervalScene()
    scene.render()
    return str(file_path)
