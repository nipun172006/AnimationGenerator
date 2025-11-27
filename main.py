from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from manim import tempconfig
import traceback
from typing import Optional

from animation_config import FunctionPlotConfig, slugify
from scenes.function_plot_scene import FunctionPlotScene
from animation_config import VectorAdditionConfig
from scenes.vector_addition_scene import VectorAdditionScene


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_function_plot(cfg: FunctionPlotConfig, outputs_dir: Path) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    video_dir = outputs_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Derive base filename from title for readability
    safe_title = slugify(cfg.title)
    base_name = f"function_plot_{safe_title}" if safe_title else "function_plot"

    # Configure Manim to render directly to outputs/videos with mp4 format
    with tempconfig({
        "media_dir": str(outputs_dir),
        "video_dir": str(video_dir),
        "format": "mp4",
        "disable_caching": True,
        "output_file": base_name,
    }):
        scene = FunctionPlotScene(cfg)
        scene.render()

    # Manim places output into video_dir/<module_name>/<quality>/<filename>.mp4 by default if not overridden.
    # With output_file set, it should be in video_dir or video_dir/quality.
    # We'll search in video_dir.
    mp4_candidates = list(video_dir.rglob(f"{base_name}.mp4"))
    if mp4_candidates:
        # Pick the most recent file
        target = max(mp4_candidates, key=lambda p: p.stat().st_mtime)
        return target

    # Fallback: return expected path
    return video_dir / f"{base_name}.mp4"


def render_vector_addition(cfg: VectorAdditionConfig, outputs_dir: Path) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    video_dir = outputs_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    safe_title = slugify(cfg.title or "vector-addition")
    base_name = f"vector_addition_{safe_title}" if safe_title else "vector_addition"

    with tempconfig({
        "media_dir": str(outputs_dir),
        "video_dir": str(video_dir),
        "format": "mp4",
        "disable_caching": True,
        "output_file": base_name,
    }):
        scene = VectorAdditionScene(cfg)
        scene.render()

    mp4_candidates = list(video_dir.rglob(f"{base_name}.mp4"))
    if mp4_candidates:
        return max(mp4_candidates, key=lambda p: p.stat().st_mtime)
    return video_dir / f"{base_name}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render educational function plots using Manim from a JSON config."
    )
    parser.add_argument("config_path", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    try:
        data = load_json(config_path)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {config_path}: {exc}") from exc

    # For STEP 1 we support only 'function_plot'
    if data.get("mode") != "function_plot":
        raise SystemExit("Unsupported mode. For STEP 1, use mode='function_plot'.")

    try:
        cfg = FunctionPlotConfig.from_dict(data)
    except Exception as exc:
        raise SystemExit(f"Invalid config: {exc}") from exc

    outputs_dir = Path("outputs")

    try:
        out_path = render_function_plot(cfg, outputs_dir)
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(f"Rendering failed: {exc}") from exc

    print(f"Rendered video: {out_path}")


if __name__ == "__main__":
    main()
