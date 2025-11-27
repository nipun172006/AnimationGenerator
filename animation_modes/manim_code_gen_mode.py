from __future__ import annotations
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from animation_config import ManimCodeGenConfig
from .base import ensure_outputs_dir

def validate_code_safety(code: str):
    """Basic safety check for generated code."""
    # This is not a sandbox, just a heuristic.
    forbidden = ["import os", "import sys", "import subprocess", "import shutil", "open(", "exec(", "eval(", "__import__"]
    for term in forbidden:
        if term in code:
            raise ValueError(f"Unsafe code detected: '{term}' is not allowed.")

def render_manim_code_gen(cfg: ManimCodeGenConfig) -> str:
    # 1. Validate Code
    validate_code_safety(cfg.code)
    
    # 2. Setup Directories
    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = "".join(x for x in (cfg.title or 'custom') if x.isalnum() or x in " -_").strip()
    file_stem = f"custom_{safe_title.replace(' ', '-')}"
    
    # 3. Create Temp Script
    with tempfile.TemporaryDirectory() as tmpdirname:
        script_path = Path(tmpdirname) / "scene.py"
        
        # Ensure imports are present
        code_content = cfg.code
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
