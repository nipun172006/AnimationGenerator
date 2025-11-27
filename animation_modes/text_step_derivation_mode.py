from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field
from .base import AnimationConfig, ensure_outputs_dir


class TextStepDerivationConfig(AnimationConfig):
    mode: Literal["text_step_derivation"] = "text_step_derivation"
    steps: List[str]
    title: Optional[str] = None


def render_text_step_derivation(cfg: TextStepDerivationConfig) -> str:
    # Minimal scene: reveal each line one by one
    from manim import Scene, VGroup, Text, FadeIn

    class DerivationScene(Scene):
        def construct(self):
            group = VGroup()
            for s in cfg.steps:
                group.add(Text(s).scale(0.6))
            group.arrange(direction=2, buff=0.3)  # UP
            for item in group:
                self.play(FadeIn(item))
                self.wait(0.5)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"text_derivation_{(cfg.title or 'steps').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename

    from manim import config
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = DerivationScene()
    scene.render()

    return str(file_path)
