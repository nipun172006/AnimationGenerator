from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class MatrixVisualizationConfig(AnimationConfig):
    mode: Literal["matrix_visualization"] = "matrix_visualization"
    matrix: Optional[List[List[Union[float, str, int]]]] = None
    # Optional second matrix for operations
    matrix_b: Optional[List[List[Union[float, str, int]]]] = None
    
    # New aliases matching User's Qwen Prompt
    matrix_A: Optional[List[List[Union[float, str, int]]]] = None
    matrix_B: Optional[List[List[Union[float, str, int]]]] = None
    
    operation: Optional[Literal["multiplication", "addition"]] = None
    
    # highlight type: row, column, diagonal, or specific cells
    highlight: Literal["row", "column", "diagonal", "cells"] = "row"
    index: int = 0
    # For 'cells' mode, list of [row, col]
    highlight_cells: Optional[List[List[int]]] = None
    duration_seconds: float = 10.0
    title: Optional[str] = None
    
    # New fields from user prompt
    highlight_color_A: Optional[str] = "BLUE"
    highlight_color_B: Optional[str] = "GREEN"
    show_intermediate_steps: bool = True


def render_matrix_visualization(cfg: MatrixVisualizationConfig) -> str:
    from manim import Scene, Text, VGroup, Create, FadeIn, FadeOut, Write, config, UP, DOWN, LEFT, RIGHT, ORIGIN, YELLOW, GREEN, RED, BLUE, SurroundingRectangle, Line, MathTex
    
    class MatrixScene(Scene):
        def create_matrix_mobject(self, matrix_data):
            rows = len(matrix_data)
            cols = len(matrix_data[0]) if rows > 0 else 0
            
            matrix_group = VGroup()
            row_groups = []
            col_groups = [VGroup() for _ in range(cols)]
            elements_map = {} # (r,c) -> Mobject
            
            for r in range(rows):
                row_group = VGroup()
                for c in range(cols):
                    val = matrix_data[r][c]
                    if isinstance(val, (int, float)):
                        txt_val = str(int(val)) if val == int(val) else f"{val:.1f}"
                    else:
                        txt_val = str(val)
                    element = Text(txt_val, font_size=36)
                    row_group.add(element)
                    col_groups[c].add(element)
                    elements_map[(r,c)] = element
                
                row_group.arrange(RIGHT, buff=0.8)
                row_groups.append(row_group)
                matrix_group.add(row_group)
            
            matrix_group.arrange(DOWN, buff=0.5)
            
            # Brackets
            if rows > 0 and cols > 0:
                height = matrix_group.height + 0.5
                bracket_left = VGroup(
                    Line([0, height/2, 0], [0, -height/2, 0]),
                    Line([0, height/2, 0], [0.2, height/2, 0]),
                    Line([0, -height/2, 0], [0.2, -height/2, 0])
                ).set_stroke(width=4).next_to(matrix_group, LEFT, buff=0.2)
                
                bracket_right = VGroup(
                    Line([0, height/2, 0], [0, -height/2, 0]),
                    Line([0, height/2, 0], [-0.2, height/2, 0]),
                    Line([0, -height/2, 0], [-0.2, -height/2, 0])
                ).set_stroke(width=4).next_to(matrix_group, RIGHT, buff=0.2)
                
                full_group = VGroup(bracket_left, matrix_group, bracket_right)
                return full_group, row_groups, col_groups, elements_map
            return matrix_group, row_groups, col_groups, elements_map

        def construct(self):
            # Timing
            total_duration = max(5.0, cfg.duration_seconds)
            
            # Title
            title = Text(cfg.title or "Matrix Visualization", font_size=40)
            title.to_edge(UP)
            self.play(Write(title), run_time=1.0)
            
            if cfg.operation == "multiplication" and cfg.matrix_b:
                self.animate_multiplication(total_duration)
            else:
                self.animate_single_matrix(total_duration)

        def animate_multiplication(self, total_duration):
            # Resolve matrices
            mat_a_data = cfg.matrix_A if cfg.matrix_A else cfg.matrix
            mat_b_data = cfg.matrix_B if cfg.matrix_B else cfg.matrix_b
            
            if not mat_a_data or not mat_b_data:
                return # Should error handle

            # Create matrices
            m1, rows1, cols1, map1 = self.create_matrix_mobject(mat_a_data)
            m2, rows2, cols2, map2 = self.create_matrix_mobject(mat_b_data)
            
            # Layout: A * B = C
            group = VGroup(m1, Text("x"), m2, Text("=")).arrange(RIGHT, buff=0.5)
            group.move_to(ORIGIN).shift(UP * 0.5)
            
            self.play(FadeIn(group), run_time=2.0)
            
            # Result matrix placeholder (empty grid first)
            # Calculate dimensions: (r1 x c1) * (r2 x c2) -> (r1 x c2)
            r1 = len(mat_a_data)
            c1 = len(mat_a_data[0])
            r2 = len(mat_b_data)
            c2 = len(mat_b_data[0])
            
            if c1 != r2:
                err = Text("Dimension Mismatch!", color=RED).next_to(group, DOWN)
                self.play(FadeIn(err))
                return

            # Animate calculation
            # We will show result matrix building up
            # For simplicity, let's just show the result matrix appearing element by element
            # But calculating the values if they are numbers
            
            result_data = [[0] * c2 for _ in range(r1)]
            # Try to calculate if numbers
            try:
                for i in range(r1):
                    for j in range(c2):
                        val = 0
                        for k in range(c1):
                            val += float(mat_a_data[i][k]) * float(mat_b_data[k][j])
                        result_data[i][j] = val
            except:
                # Fallback for symbols: just show "?"
                result_data = [["?" for _ in range(c2)] for _ in range(r1)]

            m3, rows3, cols3, map3 = self.create_matrix_mobject(result_data)
            m3.next_to(group, RIGHT, buff=0.5)
            
            # If it's too wide, scale down
            full_width = group.width + m3.width + 1.0
            if full_width > 13:
                scale = 13 / full_width
                group.scale(scale)
                m3.scale(scale)
                m3.next_to(group, RIGHT, buff=0.5 * scale)
                group.move_to(ORIGIN).shift(LEFT * m3.width/2)

            # Animate row * col -> cell
            step_time = (total_duration - 3.0) / (r1 * c2)
            
            for i in range(r1):
                for j in range(c2):
                    # Highlight row i of A
                    rect1 = SurroundingRectangle(rows1[i], color=cfg.highlight_color_A or BLUE, buff=0.1)
                    # Highlight col j of B
                    rect2 = SurroundingRectangle(cols2[j], color=cfg.highlight_color_B or GREEN, buff=0.1)
                    
                    self.play(Create(rect1), Create(rect2), run_time=step_time*0.3)
                    
                    # Show result cell
                    cell = map3[(i,j)]
                    self.play(FadeIn(cell), run_time=step_time*0.4)
                    
                    self.play(FadeOut(rect1), FadeOut(rect2), run_time=step_time*0.3)
            
            # Add brackets for result
            # (Re-create brackets for m3 since we only faded in cells)
            # Actually create_matrix_mobject returns a group with brackets if present
            # But we only accessed map3. Let's just fade in the brackets of m3 now
            if len(m3) > 1: # Has brackets
                 self.play(FadeIn(m3[0]), FadeIn(m3[2]), run_time=1.0) # Left and Right brackets
            self.wait(2.0)

        def animate_single_matrix(self, total_duration):
            mat_data = cfg.matrix_A if cfg.matrix_A else cfg.matrix
            if not mat_data:
                 mat_data = [[1,0],[0,1]] # Default
            
            m1, rows, cols, elements_map = self.create_matrix_mobject(mat_data)
            m1.move_to(ORIGIN)
            self.play(FadeIn(m1), run_time=1.0)
            
            t_highlight = max(2.0, total_duration - 3.0)
            
            # Highlight Logic
            to_highlight = []
            
            if cfg.highlight == "row":
                if 0 <= cfg.index < len(rows):
                    to_highlight.append((rows[cfg.index], YELLOW, f"Row {cfg.index}"))
            elif cfg.highlight == "column":
                if 0 <= cfg.index < len(cols):
                    to_highlight.append((cols[cfg.index], GREEN, f"Column {cfg.index}"))
            elif cfg.highlight == "diagonal":
                count = min(len(rows), len(cols))
                for i in range(count):
                    to_highlight.append((elements_map[(i,i)], RED, None))
            elif cfg.highlight == "cells" and cfg.highlight_cells:
                for r, c in cfg.highlight_cells:
                    if (r, c) in elements_map:
                        to_highlight.append((elements_map[(r,c)], RED, None))

            if to_highlight:
                step_time = t_highlight / len(to_highlight)
                for item, color, label_text in to_highlight:
                    rect = SurroundingRectangle(item, color=color, buff=0.15)
                    anims = [Create(rect)]
                    lbl = None
                    if label_text:
                        lbl = Text(label_text, font_size=24, color=color)
                        if cfg.highlight == "row":
                            lbl.next_to(rect, RIGHT)
                        else:
                            lbl.next_to(rect, DOWN)
                        anims.append(FadeIn(lbl))
                    
                    self.play(*anims, run_time=step_time * 0.5)
                    self.wait(step_time * 0.3)
                    self.play(FadeOut(rect), FadeOut(lbl) if lbl else FadeOut(VGroup()), run_time=step_time * 0.2)
            
            self.wait(1.0)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'matrix') if x.isalnum() or x in " -_").strip()
    file_stem = f"matrix_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    scene = MatrixScene()
    scene.render()
    
    return str(file_path)
