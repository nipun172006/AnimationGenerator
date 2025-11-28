from __future__ import annotations
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime
from animation_config import ManimCodeGenConfig
from .base import ensure_outputs_dir

def validate_code_safety(code: str):
    """Basic safety check for generated code."""
    # This is not a sandbox, just a heuristic.
    forbidden = ["import os", "import sys", "import subprocess", "import shutil", "open(", "exec(", "eval(", "__import__"]
    for term in forbidden:
        if term in code:
            raise ValueError(f"Unsafe code detected: '{term}' is not allowed.")

def sanitize_manim_code_for_latex(code: str) -> str:
    """
    Replace any usage of MathTex/Tex with Text, since LaTeX is not available.
    """
    # Basic replacements to avoid LaTeX-based classes
    code = code.replace("MathTex(", "Text(")
    code = code.replace("Tex(", "Text(")
    code = code.replace("TexText(", "Text(")
    return code

def sanitize_manim_code_for_arcs(code: str) -> str:
    """Prevent fragile ArcBetweenPoints usage by replacing with simple Line/Arrow if present."""
    if "ArcBetweenPoints(" in code:
        code = code.replace("ArcBetweenPoints", "Line")
    return code

def sanitize_echoed_prompt(code: str) -> str:
    """
    Remove any echoed prompt text from the beginning of the code.
    We look for the first occurrence of 'from manim import' or 'class GeneratedScene'.
    """
    # Find the start of the code
    import_idx = code.find("from manim import")
    class_idx = code.find("class GeneratedScene")
    
    start_idx = -1
    if import_idx != -1:
        start_idx = import_idx
    elif class_idx != -1:
        start_idx = class_idx
        
    if start_idx != -1:
        # Check if there's significant text before the code
        preamble = code[:start_idx].strip()
        if len(preamble) > 20: # Arbitrary threshold to detect echoed prompt
            print("WARNING: Detected echoed prompt in code. Stripping preamble.")
            return code[start_idx:]
            
    return code

def sanitize_axis_labels(code: str) -> str:
    """
    Comment out usage of .get_axis_labels() which triggers LaTeX.
    """
    if ".get_axis_labels(" in code:
        print("WARNING: Disabling .get_axis_labels() to prevent LaTeX crash.")
        # We replace it with a pass-through or comment out the line.
        # Simple replace: comment it out.
        code = code.replace(".get_axis_labels(", "# .get_axis_labels(")
    return code

def sanitize_axes_numbers(code: str) -> str:
    """
    Ensure that no Axes/NumberLine uses include_numbers=True,
    because that triggers LaTeX-based DecimalNumber labels.
    """
    # Handle both double-quoted and single-quoted cases
    code = code.replace('"include_numbers": True', '"include_numbers": False')
    code = code.replace("'include_numbers': True", "'include_numbers': False")
    code = code.replace('"include_numbers":True', '"include_numbers": False')
    code = code.replace("'include_numbers':True", "'include_numbers': False")
    
    if '"include_numbers": False' in code or "'include_numbers': False" in code:
        print("INFO: Forced include_numbers=False to prevent LaTeX crashes.")
    
    return code

def sanitize_matrix_code(code: str) -> str:
    """
    Fix common matrix multiplication code errors.
    """
    # Fix: Text(...).scale(...) being called on string
    # Replace: Text(str(...).scale(...)) with Text(str(...), font_size=24)
    import re
    
    # Fix the pattern: Text(str(...)).scale(0.5) -> Text(str(...), font_size=24)
    code = re.sub(
        r'Text\(str\(([^)]+)\)\)\.scale\([^)]+\)',
        r'Text(str(\1), font_size=24)',
        code
    )
    
    # Fix: result[i].set_text(...) which is deprecated
    # Replace with proper Text object creation
    if '.set_text(' in code:
        print("WARNING: Detected .set_text() usage. Replacing with proper Text object creation.")
        # This is complex to fix with regex, so let's add a comment
        code = code.replace('.set_text(', '# .set_text(  # DEPRECATED - ')
    
    # Fix: Transform(result[i], Text(str(...).scale(...)))
    # The issue is str(...).scale() - scale is not a string method
    code = re.sub(
        r'Transform\(([^,]+),\s*Text\(str\(([^)]+)\)\.scale\([^)]+\)\)\)',
        r'# FIXED: Transform(\1, Text(str(\2), font_size=24))',
        code
    )
    
    return code

def render_manim_code_gen(cfg: ManimCodeGenConfig) -> str:
    # 1. Validate Code
    validate_code_safety(cfg.code)
    
    # Sanitize echoed prompt FIRST
    code_content = sanitize_echoed_prompt(cfg.code)
    
    # Sanitize for LaTeX (No LaTeX environment available)
    code_content = sanitize_manim_code_for_latex(code_content)
    
    # Sanitize for Arcs
    code_content = sanitize_manim_code_for_arcs(code_content)
    
    # Sanitize Axis Labels (LaTeX trigger)
    code_content = sanitize_axis_labels(code_content)
    
    # Sanitize Axes Numbers (LaTeX trigger from include_numbers=True)
    code_content = sanitize_axes_numbers(code_content)
    
    # Sanitize Matrix Code (fix common errors)
    code_content = sanitize_matrix_code(code_content)
    
    # Check for plot_point usage (warning only)
    if ".plot_point(" in code_content:
        print("WARNING: Code uses .plot_point(), which might fail. SLM should have been instructed to use coords_to_point.")

    # Check for potential 2D Arrow coordinates (heuristic)
    if "Arrow(start=(" in code_content or "Arrow(start=[" in code_content:
        print("WARNING: Potential 2D coordinates detected in Arrow. SLM should use 3D vectors.")
    
    # 2. Setup Directories
    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'custom') if x.isalnum() or x in " -_").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"custom_{safe_title.replace(' ', '-')}_{timestamp}"
    
    # 3. Create Temp Script
    with tempfile.TemporaryDirectory() as tmpdirname:
        script_path = Path(tmpdirname) / "scene.py"
        
        # Ensure imports are present
        if "from manim import *" not in code_content:
            code_content = "from manim import *\n" + code_content
            
        with open(script_path, "w") as f:
            f.write(code_content)
            
        # 4. Run Manim
        # We run in a subprocess.
        # Command: manim -ql --media_dir <outputs_dir> -o <file_stem> <script_path> GeneratedScene
        
        cmd = [
            sys.executable, "-m", "manim",
            "-ql", # Low quality for speed
            "--media_dir", str(outputs_dir),
            "-o", file_stem,
            str(script_path),
            "GeneratedScene"
        ]
        
        # Run
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Manim rendering timed out (limit: 120s).")
            
        if result.returncode != 0:
            # Try to extract the error message
            error_msg = result.stderr
            if "ModuleNotFoundError" in error_msg:
                error_msg += "\n(Only standard libraries + manim are supported)"
            raise RuntimeError(f"Manim execution failed:\n{error_msg}")
            
        # 5. Locate Output File
        # Manim output structure: <media_dir>/videos/<script_name>/480p15/<file_stem>.mp4
        # script_name is "scene"
        video_path = Path(outputs_dir) / "videos" / "scene" / "480p15" / f"{file_stem}.mp4"
        
        if not video_path.exists():
            # Fallback search
            found = list(Path(outputs_dir).rglob(f"{file_stem}.mp4"))
            if found:
                video_path = found[0]
            else:
                raise RuntimeError(f"Manim finished but video file not found at expected path: {video_path}")
        
        # 6. Move to final destination
        final_path = out_dir / f"{file_stem}.mp4"
        if video_path != final_path:
            shutil.move(str(video_path), str(final_path))
            
    return str(final_path)
