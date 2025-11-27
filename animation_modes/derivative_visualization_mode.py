from __future__ import annotations
from pathlib import Path
from manim import *
from animation_config import DerivativeVisualizationConfig, build_function_callable
from .base import ensure_outputs_dir

def render_derivative_visualization(cfg: DerivativeVisualizationConfig) -> str:
    class DerivativeScene(Scene):
        def construct(self):
            # 1. Setup Function
            try:
                func = build_function_callable(cfg.function_expression)
            except Exception as e:
                error_text = Text(f"Error: {e}", color=RED).scale(0.5)
                self.add(error_text)
                return

            # 2. Setup Axes
            # Determine y range dynamically or fixed?
            # Let's sample a few points to guess y range
            x_vals = np.linspace(cfg.x_min, cfg.x_max, 20)
            y_vals = [func(x) for x in x_vals]
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
                if abs(val) < 0.1: continue # Skip 0
                t = Text(f"{val:.1f}", font_size=16).next_to(axes.c2p(val, 0), DOWN)
                self.add(t)
                
            y_step = (y_max - y_min) / 4
            for i in range(5):
                val = y_min + i * y_step
                if abs(val) < 0.1: continue
                t = Text(f"{val:.1f}", font_size=16).next_to(axes.c2p(0, val), LEFT)
                self.add(t)

            # 3. Plot Function
            graph = axes.plot(func, color=WHITE)
            # Manual graph label
            graph_label = Text(f"f(x)={cfg.function_expression}", font_size=24, color=WHITE).to_corner(UL)

            self.play(Create(axes), run_time=1.5)
            self.play(Create(graph), Write(graph_label), run_time=2.0)

            # 4. Tangent Line at x0
            # Calculate slope numerically
            h = 0.001
            slope = (func(cfg.x0 + h) - func(cfg.x0 - h)) / (2 * h)
            
            # Point
            p0 = axes.c2p(cfg.x0, func(cfg.x0))
            dot = Dot(p0, color=YELLOW)
            
            # Tangent line
            # y - y0 = m(x - x0) => y = m(x - x0) + y0
            def tangent_func(x):
                return slope * (x - cfg.x0) + func(cfg.x0)
            
            tangent_line = axes.plot(tangent_func, x_range=[cfg.x_min, cfg.x_max], color=YELLOW)
            
            self.play(FadeIn(dot), Create(tangent_line), run_time=1.5)
            
            # Label slope
            slope_text = Text(f"Slope: {slope:.2f}", font_size=24, color=YELLOW).next_to(dot, UP)
            self.play(Write(slope_text))

            # 5. Derivative Curve (Optional)
            if cfg.show_derivative_curve:
                def deriv_func(x):
                    return (func(x + h) - func(x - h)) / (2 * h)
                
                deriv_graph = axes.plot(deriv_func, color=RED)
                deriv_label = Text("f'(x)", font_size=24, color=RED).next_to(deriv_graph.get_end(), RIGHT)
                
                self.play(TransformFromCopy(graph, deriv_graph), Write(deriv_label), run_time=2.0)

            # Title
            if cfg.title:
                title = Text(cfg.title).to_edge(UP)
                self.play(FadeIn(title))

            self.wait(2)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'derivative') if x.isalnum() or x in " -_").strip()
    file_stem = f"derivative_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    scene = DerivativeScene()
    scene.render()
    return str(file_path)
