from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "sample_configs"
MAIN = ROOT / "main.py"


def run_config(cfg_path: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(MAIN), str(cfg_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_output_path(stdout: str) -> str | None:
    for line in stdout.strip().splitlines()[::-1]:
        if line.startswith("Rendered video: "):
            return line.split("Rendered video: ", 1)[1].strip()
    return None


def main() -> None:
    if not SAMPLES_DIR.exists():
        print(f"Samples not found: {SAMPLES_DIR}")
        sys.exit(1)

    json_files = sorted(SAMPLES_DIR.glob("*.json"))
    if not json_files:
        print("No sample configs found.")
        sys.exit(1)

    print(f"Found {len(json_files)} sample configs. Rendering...\n")

    failures: list[tuple[Path, str]] = []
    for cfg in json_files:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append((cfg, f"Invalid JSON: {exc}"))
            continue
        print(f"→ {cfg.name}: {data.get('title', data.get('function_expression', 'unnamed'))}")
        code, out, err = run_config(cfg)
        path = parse_output_path(out)
        if code == 0 and path:
            print(f"  ✓ Rendered: {path}\n")
        else:
            print(f"  ✗ Failed ({code}).")
            if err:
                print(err)
            failures.append((cfg, err or "Unknown error"))
            print()

    if failures:
        print("Summary: Some renders failed:")
        for cfg, msg in failures:
            print(f" - {cfg.name}: {msg.splitlines()[-1] if msg else 'error'}")
        sys.exit(2)

    print("All samples rendered successfully.")


if __name__ == "__main__":
    main()
