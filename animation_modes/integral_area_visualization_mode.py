from __future__ import annotations
from pathlib import Path
from manim import *
from animation_config import IntegralAreaVisualizationConfig, build_function_callable
from .base import ensure_outputs_dir

def render_integral_area_visualization(cfg: IntegralAreaVisualizationConfig) -> str:
    class IntegralScene(Scene):
        def construct(self):
            # 1. Setup Function
            try:
                func = build_function_callable(cfg.function_expression)
            except Exception as e:
                error_text = Text(f"Error: {e}", color=RED).scale(0.5)
                self.add(error_text)
                return

            # 2. Setup Axes
            # Default range -5 to 5 if a,b are small, else expand
            x_padding = (cfg.b - cfg.a) * 0.5 + 1
            x_min = min(-5, cfg.a - x_padding)
            x_max = max(5, cfg.b + x_padding)
            
            # Y range guess
            x_vals = np.linspace(x_min, x_max, 20)
            y_vals = [func(x) for x in x_vals]
            y_min_calc, y_max_calc = min(y_vals), max(y_vals)
            y_padding = (y_max_calc - y_min_calc) * 0.2 + 1
            y_min = y_min_calc - y_padding
            y_max = y_max_calc + y_padding

            axes = Axes(
                x_range=[x_min, x_max, (x_max - x_min) / 8],
                y_range=[y_min, y_max, (y_max - y_min) / 6],
                axis_config={"color": BLUE, "include_numbers": False},
            )
            
            # Manually add axis numbers
            x_step = (x_max - x_min) / 4
            for i in range(5):
                val = x_min + i * x_step
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
            graph = axes.plot(func, color=WHITE)
            
            self.play(Create(axes), run_time=1.5)
            self.play(Create(graph), run_time=1.5)

            # 4. Riemann Rectangles
            # Map method to input_sample_type
            method_map = {"left": "left", "right": "right", "midpoint": "center"}
            sample_type = method_map.get(cfg.method, "center")
            
            rects = axes.get_riemann_rectangles(
                graph,
                x_range=[cfg.a, cfg.b],
                dx=(cfg.b - cfg.a) / cfg.num_rectangles,
                input_sample_type=sample_type,
                stroke_width=1,
                fill_opacity=0.5,
                color=BLUE
            )
            
            self.play(Create(rects), run_time=3.0)
            
            # 5. Exact Area (Shaded)
            area = axes.get_area(graph, x_range=[cfg.a, cfg.b], color=GREEN, opacity=0.3)
            self.play(FadeIn(area), run_time=1.5)

            # 6. Label
            if cfg.show_exact_area_label:
                # Calculate approximate area
                # Simple numerical integration
                dx = 0.01
                xs = np.arange(cfg.a, cfg.b, dx)
                approx_area = sum(func(x) * dx for x in xs)
                
                label = Text(f"Area ≈ {approx_area:.2f}", font_size=24).to_corner(UR)
                self.play(Write(label))

            # Title
            if cfg.title:
                title = Text(cfg.title).to_edge(UP)
                self.play(FadeIn(title))

            self.wait(2)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'integral') if x.isalnum() or x in " -_").strip()
    file_stem = f"integral_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    scene = IntegralScene()
    scene.render()
    return str(file_path)
