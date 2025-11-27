from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, BaseModel
from .base import AnimationConfig, ensure_outputs_dir


class ExplainerSection(BaseModel):
    heading: str
    bullet_points: List[str]


class GenericExplainerConfig(AnimationConfig):
    mode: Literal["generic_explainer"] = "generic_explainer"
    sections: List[ExplainerSection]
    title: Optional[str] = None


def render_generic_explainer(cfg: GenericExplainerConfig) -> str:
    from manim import Scene, Text, VGroup, FadeIn, config

    class ExplainerScene(Scene):
        def construct(self):
            # Title screen
            title_text = Text(cfg.title or "Explainer").scale(0.8)
            self.play(FadeIn(title_text))
            self.wait(0.8)
            self.play(title_text.animate.scale(0.6).to_edge(1))  # UP

            for section in cfg.sections:
                heading = Text(section.heading).scale(0.6).to_edge(1)  # UP
                bullets_group = VGroup()
                for bp in section.bullet_points:
                    bullets_group.add(Text("• " + bp).scale(0.4))
                bullets_group.arrange(direction=2, aligned_edge=1, buff=0.25)  # UP alignment
                bullets_group.next_to(heading, 2, buff=0.4)  # DOWN
                self.play(FadeIn(heading))
                for b in bullets_group:
                    self.play(FadeIn(b))
                    self.wait(0.2)
                self.wait(0.5)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_basename = f"generic_explainer_{(cfg.title or 'explainer').replace(' ', '-')}.mp4"
    file_path = out_dir / file_basename
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    scene = ExplainerScene()
    scene.render()
    return str(file_path)
