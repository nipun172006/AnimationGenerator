from __future__ import annotations
from pathlib import Path
from manim import *
from animation_config import LimitVisualizationConfig, build_function_callable
from .base import ensure_outputs_dir

def render_limit_visualization(cfg: LimitVisualizationConfig) -> str:
    class LimitScene(Scene):
        def construct(self):
            # 1. Setup Function
            try:
                func = build_function_callable(cfg.function_expression)
            except Exception as e:
                error_text = Text(f"Error: {e}", color=RED).scale(0.5)
                self.add(error_text)
                return

            # 2. Setup Axes
            # Y range guess
            x_vals = np.linspace(cfg.x_min, cfg.x_max, 50)
            # Filter out potential singularities near x0 if function is undefined there
            # But build_function_callable might raise error or return inf.
            # We'll just try/except inside list comp
            y_vals = []
            for x in x_vals:
                try:
                    val = func(x)
                    if np.isfinite(val):
                        y_vals.append(val)
                except: pass
            
            if not y_vals:
                y_min, y_max = -5, 5
            else:
                y_min_calc, y_max_calc = min(y_vals), max(y_vals)
                y_padding = (y_max_calc - y_min_calc) * 0.2 + 1
                y_min = y_min_calc - y_padding
                y_max = y_max_calc + y_padding

            axes = Axes(
                x_range=[cfg.x_min, cfg.x_max, (cfg.x_max - cfg.x_min) / 8],
                y_range=[y_min, y_max, (y_max - y_min) / 6],
                axis_config={"color": BLUE, "include_numbers": False},
            )
            
            # Manually add axis numbers
            x_step = (cfg.x_max - cfg.x_min) / 4
            for i in range(5):
                val = cfg.x_min + i * x_step
                if abs(val) < 0.1: continue
                t = Text(f"{val:.1f}", font_size=16).next_to(axes.c2p(val, 0), DOWN)
                self.add(t)
                
            y_step = (y_max - y_min) / 4
            for i in range(5):
                val = y_min + i * y_step
                if abs(val) < 0.1: continue
                t = Text(f"{val:.1f}", font_size=16).next_to(axes.c2p(0, val), LEFT)
                self.add(t)

            # 3. Plot Function
            # Handle discontinuity at x0? Manim plots usually handle it by skipping or drawing steep lines.
            # We can plot in two segments if x0 is in range
            graph = axes.plot(func, color=WHITE, discontinuities=[cfg.x0], dt=0.01)
            
            self.play(Create(axes), run_time=1.5)
            self.play(Create(graph), run_time=1.5)

            # 4. Limit Approach Animation
            # Vertical line at x0
            vline = DashedLine(
                start=axes.c2p(cfg.x0, y_min),
                end=axes.c2p(cfg.x0, y_max),
                color=GRAY
            )
            self.play(Create(vline))

            # Approaching dots
            # Left approach
            if cfg.show_left_right:
                dot_left = Dot(color=YELLOW)
                
                # Value tracker for x
                x_tracker_left = ValueTracker(cfg.x_min)
                
                dot_left.add_updater(lambda d: d.move_to(axes.c2p(x_tracker_left.get_value(), func(x_tracker_left.get_value()))))
                
                self.add(dot_left)
                self.play(x_tracker_left.animate.set_value(cfg.x0 - 0.05), run_time=3.0)
                
                # Right approach
                dot_right = Dot(color=GREEN)
                x_tracker_right = ValueTracker(cfg.x_max)
                
                dot_right.add_updater(lambda d: d.move_to(axes.c2p(x_tracker_right.get_value(), func(x_tracker_right.get_value()))))
                
                self.add(dot_right)
                self.play(x_tracker_right.animate.set_value(cfg.x0 + 0.05), run_time=3.0)

            # Title
            if cfg.title:
                title = Text(cfg.title).to_edge(UP)
                self.play(FadeIn(title))

            self.wait(2)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'limit') if x.isalnum() or x in " -_").strip()
    file_stem = f"limit_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    scene = LimitScene()
    scene.render()
    return str(file_path)
