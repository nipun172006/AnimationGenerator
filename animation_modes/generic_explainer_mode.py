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
    duration_seconds: float = 12.0


def render_generic_explainer(cfg: GenericExplainerConfig) -> str:
    from manim import Scene, Text, VGroup, FadeIn, FadeOut, Write, config, UP, DOWN, LEFT, ORIGIN
    
    class ExplainerScene(Scene):
        def construct(self):
            # 1. Calculate timing
            total_duration = cfg.duration_seconds
            # Steps: 1 for title intro + 1 per section
            num_steps = 1 + len(cfg.sections)
            step_time = total_duration / max(1, num_steps)
            
            # 2. Title Phase
            frame_width = self.camera.frame_width
            frame_height = self.camera.frame_height
            
            title_text = Text(cfg.title or "Explainer", font_size=48)
            # Ensure title fits
            if title_text.width > frame_width * 0.9:
                title_text.scale_to_fit_width(frame_width * 0.9)
            title_text.to_edge(UP)
            
            # Intro animation
            self.play(Write(title_text), run_time=step_time * 0.5)
            self.wait(step_time * 0.5)
            
            current_group = None
            
            # 3. Sections Phase
            for section in cfg.sections:
                # Prepare content
                heading = Text(section.heading, font_size=36, color="#4AF626")
                # Scale heading
                if heading.width > frame_width * 0.85:
                    heading.scale_to_fit_width(frame_width * 0.85)
                
                bullets = []
                for bp in section.bullet_points:
                    b = Text("• " + bp, font_size=28)
                    # Scale bullet if too wide
                    if b.width > frame_width * 0.85:
                        b.scale_to_fit_width(frame_width * 0.85)
                    bullets.append(b)
                
                bullet_group = VGroup(*bullets)
                bullet_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
                
                # Layout
                heading.next_to(title_text, DOWN, buff=0.5)
                bullet_group.next_to(heading, DOWN, buff=0.4)
                
                # Check vertical overflow
                content_group = VGroup(heading, bullet_group)
                available_height = frame_height - title_text.height - 1.5
                if content_group.height > available_height:
                    content_group.scale_to_fit_height(available_height)
                    # Re-position after scaling
                    content_group.next_to(title_text, DOWN, buff=0.5)
                
                # Transition: Fade out old, Fade in new
                if current_group:
                    self.play(FadeOut(current_group), run_time=step_time * 0.2)
                
                self.play(FadeIn(content_group), run_time=step_time * 0.3)
                
                # Wait for reading
                self.wait(step_time * 0.5)
                
                current_group = content_group
            
            # End
            if current_group:
                self.play(FadeOut(current_group), run_time=1.0)
            self.play(FadeOut(title_text), run_time=0.5)

    outputs_dir = ensure_outputs_dir()
    out_dir = Path(outputs_dir, "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    safe_title = "".join(x for x in (cfg.title or 'explainer') if x.isalnum() or x in " -_").strip()
    file_stem = f"generic_explainer_{safe_title.replace(' ', '-')}"
    file_path = out_dir / f"{file_stem}.mp4"
    
    config.media_dir = str(outputs_dir)
    config.video_dir = str(out_dir)
    config.output_file = file_stem
    
    # Render
    scene = ExplainerScene()
    scene.render()
    
    return str(file_path)
