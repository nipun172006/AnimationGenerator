from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class MatrixMultDetailedConfig(AnimationConfig):
    mode: Literal["matrix_mult_detailed"] = "matrix_mult_detailed"
    title: Optional[str] = "4x4 Matrix Multiplication - Step by Step"
    duration_seconds: float = 30.0


def render_matrix_mult_detailed(cfg: MatrixMultDetailedConfig) -> str:
    from manim import Scene, Text, VGroup, Create, FadeIn, FadeOut, Write, config as manim_config, UP, DOWN, LEFT, RIGHT, ORIGIN, YELLOW, GREEN, RED, BLUE, WHITE, SurroundingRectangle
    import subprocess
    import tempfile
    import sys
    from pathlib import Path
    
    # Define sample 4x4 matrices
    A_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B_data = [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]
    
    code = f'''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Define 4x4 matrices
        A_data = {A_data}
        B_data = {B_data}
        
        # Create matrix A
        matrix_A = VGroup()
        for i in range(4):
            row = VGroup()
            for j in range(4):
                cell = Text(str(A_data[i][j]), font_size=24)
                row.add(cell)
            row.arrange(RIGHT, buff=0.3)
            matrix_A.add(row)
        matrix_A.arrange(DOWN, buff=0.3)
        
        # Create matrix B
        matrix_B = VGroup()
        for i in range(4):
            row = VGroup()
            for j in range(4):
                cell = Text(str(B_data[i][j]), font_size=24)
                row.add(cell)
            row.arrange(RIGHT, buff=0.3)
            matrix_B.add(row)
        matrix_B.arrange(DOWN, buff=0.3)
        
        # Create result matrix (initially zeros)
        result_matrix = VGroup()
        result_data = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            row = VGroup()
            for j in range(4):
                cell = Text("0", font_size=24, color=BLUE)
                row.add(cell)
            row.arrange(RIGHT, buff=0.3)
            result_matrix.add(row)
        result_matrix.arrange(DOWN, buff=0.3)
        
        # Position matrices horizontally
        matrix_A.shift(LEFT * 4.5)
        matrix_B.next_to(matrix_A, RIGHT, buff=1.5)
        result_matrix.next_to(matrix_B, RIGHT, buff=1.5)
        
        # Add labels
        label_A = Text("A", font_size=32).next_to(matrix_A, UP)
        label_B = Text("B", font_size=32).next_to(matrix_B, UP)
        label_C = Text("C = A × B", font_size=32).next_to(result_matrix, UP)
        
        # Show matrices
        self.play(FadeIn(matrix_A), FadeIn(label_A))
        self.play(FadeIn(matrix_B), FadeIn(label_B))
        self.play(FadeIn(result_matrix), FadeIn(label_C))
        self.wait(1)
        
        # Compute only first 2x2 cells (to keep animation short)
        for i in range(2):
            for j in range(2):
                # Highlight row in A and column in B
                row_rect = SurroundingRectangle(matrix_A[i], color=YELLOW, buff=0.1)
                col_cells = VGroup(*[matrix_B[k][j] for k in range(4)])
                col_rect = SurroundingRectangle(col_cells, color=GREEN, buff=0.1)
                
                self.play(Create(row_rect), Create(col_rect))
                
                # Calculate dot product
                dot_product_sum = 0
                computation_text = VGroup()
                
                for k in range(4):
                    a_val = A_data[i][k]
                    b_val = B_data[k][j]
                    product = a_val * b_val
                    dot_product_sum += product
                    
                    # Show multiplication
                    mult_text = Text(f"{{a_val}}×{{b_val}}={{product}}", font_size=18)
                    computation_text.add(mult_text)
                
                computation_text.arrange(DOWN, buff=0.2)
                computation_text.next_to(result_matrix, DOWN, buff=0.5)
                
                self.play(FadeIn(computation_text))
                self.wait(0.3)
                
                # Show sum
                sum_text = Text(f"Sum = {{dot_product_sum}}", font_size=20, color=RED)
                sum_text.next_to(computation_text, DOWN, buff=0.3)
                self.play(Write(sum_text))
                self.wait(0.3)
                
                # Update result matrix
                result_data[i][j] = dot_product_sum
                new_cell = Text(str(dot_product_sum), font_size=24, color=RED)
                new_cell.move_to(result_matrix[i][j])
                
                self.play(FadeOut(result_matrix[i][j]))
                result_matrix[i].remove(result_matrix[i][j])
                result_matrix[i].add(new_cell)
                self.play(FadeIn(new_cell))
                
                # Clean up
                self.play(
                    FadeOut(row_rect), 
                    FadeOut(col_rect),
                    FadeOut(computation_text),
                    FadeOut(sum_text)
                )
                
                self.wait(0.2)
        
        self.wait(2)
'''
    
    # Render using Manim
    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "matrix_mult.py"
        
        with open(script_path, "w") as f:
            f.write(code)
        
        # Run Manim
        cmd = [
            sys.executable, "-m", "manim",
            "-ql",  # Low quality for speed
            "--media_dir", str(outputs_dir),
            "-o", "matrix_mult_detailed",
            str(script_path),
            "GeneratedScene"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"Manim execution failed:\n{result.stderr}")
        
        # Find the output file
        video_path = Path(outputs_dir) / "videos" / "matrix_mult" / "480p15" / "matrix_mult_detailed.mp4"
        
        if not video_path.exists():
            # Fallback search
            found = list(Path(outputs_dir).rglob("matrix_mult_detailed.mp4"))
            if found:
                video_path = found[0]
            else:
                raise RuntimeError(f"Video file not found at expected path: {video_path}")
        
        # Move to final destination
        final_path = out_dir / "matrix_mult_detailed.mp4"
        import shutil
        if video_path != final_path:
            shutil.copy(str(video_path), str(final_path))
        
        return str(final_path)
