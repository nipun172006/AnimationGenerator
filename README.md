# Text-Driven Educational Animation Video Generator (STEP 1)

This is a minimal Python + Manim skeleton implementing a single animation mode: `function_plot`.

It reads a JSON config, converts it into a typed config object, and renders a video plotting a mathematical function using Manim.

## Prerequisites

- Python 3.10+
- macOS: Ensure FFmpeg is installed (required by Manim)

```bash
# macOS
brew install ffmpeg
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run (STEP 1)

Use the provided sample config:

```bash
python main.py sample_configs/function_plot_sine.json
```

On success, the rendered MP4 will be placed under the `outputs/` directory (Manim may create nested subfolders). The console will print the final path (e.g., `outputs/videos/.../function_plot_sine-wave-with-maxima-and-minima.mp4`).

### More Examples

Try a few additional presets:

```bash
python main.py sample_configs/function_plot_cosine.json
python main.py sample_configs/function_plot_quadratic.json
python main.py sample_configs/function_plot_gaussian.json
python main.py sample_configs/function_plot_abs.json
python main.py sample_configs/function_plot_tanh.json
python main.py sample_configs/function_plot_sqrt.json
```

These cover periodic functions, polynomials, smooth decays, non-differentiable corners, saturating functions, and domain-restricted roots.

## JSON Config Schema (function_plot)

Example:

```json
{
  "mode": "function_plot",
  "function_expression": "sin(x)",
  "x_min": -6.28,
  "x_max": 6.28,
  "y_min": -2.0,
  "y_max": 2.0,
  "duration_seconds": 8,
  "title": "Sine Wave with Maxima and Minima"
}
```

- `function_expression`: A math expression string using allowed functions like `sin`, `cos`, `exp`, `log`, `sqrt`, and constants `pi` and `e`. Use `x` as the variable. `^` is supported as power (converted to `**`).
- Ranges (`x_min`, `x_max`, `y_min`, `y_max`) define the axes.
- `duration_seconds` controls how long the line draw animation takes overall.
- `title` appears at the top of the scene and is used to derive the output filename.

## Project Structure

- `main.py` — CLI entrypoint: loads JSON, validates, renders scene.
- `animation_config.py` — Dataclass for `FunctionPlotConfig`, safe math parsing, utility helpers.
- `scenes/function_plot_scene.py` — Manim `FunctionPlotScene` that plots the function.
- `sample_configs/function_plot_sine.json` — Example config.
- `outputs/` — Rendered files directory.

## Notes

- This is intentionally minimal to satisfy STEP 1 of the hackathon task. The architecture is designed to extend with additional modes (vectors, geometry, algorithms) in future steps.
- For LaTeX labels or more advanced text, additional system dependencies may be required by Manim. For STEP 1, we use `Text` to avoid LaTeX.
