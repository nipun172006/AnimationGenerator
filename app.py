from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Dict
import os
import json

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from animation_config import FunctionPlotConfig
from animation_config import VectorAdditionConfig, Vector2D
from main import render_function_plot, render_vector_addition

# New registry-based rendering
from animation_modes import render_animation_from_mode

app = FastAPI(title="Educational Animation API", version="0.1.0")

# CORS: allow local frontend origins (Vite dev server and localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FunctionPlotRequest(BaseModel):
    """Request body for function plot rendering.

    Mirrors the STEP 1 JSON format. This model is kept independent of the
    dataclass to ensure FastAPI input validation and OpenAPI generation.
    """

    mode: str = Field("function_plot", description="Animation mode; must be 'function_plot'")
    function_expression: str = Field(..., description="Math expression in terms of x, e.g., 'sin(x)' or 'x**2'")
    x_min: float = Field(..., description="Minimum x value")
    x_max: float = Field(..., description="Maximum x value")
    y_min: float = Field(..., description="Minimum y value")
    y_max: float = Field(..., description="Maximum y value")
    duration_seconds: float = Field(..., gt=0, description="Total animation duration in seconds")
    title: str = Field(..., description="Title shown atop the scene and used for the output filename")


class RenderResponse(BaseModel):
    status: str
    output_video_path: Optional[str] = None
    message: Optional[str] = None


@app.get("/health", response_model=dict)
def health() -> dict[str, str]:
    """Basic health-check endpoint."""
    return {"status": "ok"}


@app.get("/health/llm", response_model=dict)
def health_llm() -> Dict[str, Any]:
    """Report LLM enhancer configuration without exposing secrets.

    Returns which enhancer is configured (gemini/openai/none), whether the
    relevant API keys are present, and the model names if set.
    """
    gemini_key_present = bool(os.getenv("GEMINI_API_KEY"))
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))

    enhancer = "none"
    if gemini_key_present:
        enhancer = "gemini"
    elif openai_key_present:
        enhancer = "openai"

    data: Dict[str, Any] = {
        "status": "ok",
        "enhancer": enhancer,
        "gemini_configured": gemini_key_present,
        "openai_configured": openai_key_present,
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-flash-latest" if gemini_key_present else None),
        "openai_model": os.getenv("OPENAI_MODEL") if openai_key_present else None,
    }

    # Package availability (best-effort, no import side effects if missing)
    try:
        import google.generativeai  # type: ignore
        data["gemini_package"] = True
    except Exception:
        data["gemini_package"] = False
    try:
        import openai  # type: ignore
        data["openai_package"] = True
    except Exception:
        data["openai_package"] = False

    return data


@app.post("/render/function_plot", response_model=RenderResponse)
def render_function_plot_endpoint(req: FunctionPlotRequest) -> RenderResponse:
    """Render a function plot animation using Manim and return output path.

    For now, this wraps only the 'function_plot' mode. Future modes will add
    separate endpoints or a generic dispatcher.
    """
    # Validate mode explicitly
    if req.mode != "function_plot":
        raise HTTPException(status_code=400, detail="Unsupported mode; expected 'function_plot'.")

    # Build the typed config from the request
    try:
        cfg = FunctionPlotConfig.from_dict(req.dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config: {exc}")

    # Render via registry helper to keep behavior consistent
    try:
        out_path = render_animation_from_mode("function_plot", req.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


class VectorAdditionRequest(BaseModel):
    mode: str = Field("vector_addition", description="Must be 'vector_addition'")
    vectors: List[Vector2D]
    show_resultant: bool = True
    show_tip_to_tail: bool = True
    title: Optional[str] = None


@app.post("/render/vector_addition", response_model=RenderResponse)
def render_vector_addition_endpoint(req: VectorAdditionRequest) -> RenderResponse:
    if req.mode != "vector_addition":
        raise HTTPException(status_code=400, detail="Unsupported mode; expected 'vector_addition'.")

    try:
        cfg = VectorAdditionConfig(**req.dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config: {exc}")

    try:
        out_path = render_animation_from_mode("vector_addition", cfg.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")

    return RenderResponse(status="ok", output_video_path=str(out_path))


class AnyModeRequest(BaseModel):
    mode: str
    payload: Dict[str, Any]


@app.post("/render/any_mode", response_model=RenderResponse)
def render_any_mode_endpoint(req: AnyModeRequest) -> RenderResponse:
    try:
        out_path = render_animation_from_mode(req.mode, req.payload)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


# ================================
# SLM (Ollama) integration section
# ================================

OLLAMA_URL = "http://localhost:11434/api/generate"
# Update to match locally available model tag seen in /api/tags
OLLAMA_MODEL = "gemma3:4b"


class InstructionRequest(BaseModel):
    prompt: str


class InstructionResponse(BaseModel):
    status: str  # "ok" or "error"
    mode: Optional[str] = None  # "function_plot" | "vector_addition" when ok
    instructions: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    enhanced_prompt: Optional[str] = None  # Added: cloud LLM-enhanced plain-English text
    enhanced_source: Optional[str] = None  # 'gemini' | 'openai' | 'fallback'


def _build_system_prompt() -> str:
    # Expanded system prompt including new schemas; still enforces single JSON object.
    return (
        "You are a STRICT JSON generator for educational animation instructions.\n"
        "Output EXACTLY ONE JSON object matching ONE schema below. No extra text. No markdown.\n"
        "Schemas:\n\n"
        "function_plot:\n"
        "{\n  \"mode\": \"function_plot\",\n  \"function_expression\": \"string\",\n  \"x_min\": number,\n  \"x_max\": number,\n  \"y_min\": number,\n  \"y_max\": number,\n  \"duration_seconds\": number,\n  \"title\": \"string\"\n}\n\n"
        "vector_addition:\n"
        "{\n  \"mode\": \"vector_addition\",\n  \"vectors\": [ { \"label\": \"string\", \"x\": number, \"y\": number } ],\n  \"show_resultant\": boolean,\n  \"show_tip_to_tail\": boolean,\n  \"title\": \"string\"\n}\n\n"
        "bubble_sort_visualization:\n"
        "{\n  \"mode\": \"bubble_sort_visualization\",\n  \"array\": [ number ],\n  \"duration_seconds\": number,\n  \"title\": \"string\"\n}\n\n"
        "parametric_plot:\n"
        "{\n  \"mode\": \"parametric_plot\",\n  \"x_expression\": \"string\",\n  \"y_expression\": \"string\",\n  \"t_min\": number,\n  \"t_max\": number,\n  \"duration_seconds\": number,\n  \"title\": \"string\"\n}\n\n"
        "geometry_construction:\n"
        "{\n  \"mode\": \"geometry_construction\",\n  \"description_steps\": [ \"string\" ],\n  \"title\": \"string\"\n}\n\n"
        "matrix_visualization:\n"
        "{\n  \"mode\": \"matrix_visualization\",\n  \"matrix\": [ [ number ] ],\n  \"highlight\": \"row|column\",\n  \"index\": number,\n  \"title\": \"string\"\n}\n\n"
        "text_step_derivation:\n"
        "{\n  \"mode\": \"text_step_derivation\",\n  \"steps\": [ \"string\" ],\n  \"title\": \"string\"\n}\n\n"
        "number_line_interval:\n"
        "{\n  \"mode\": \"number_line_interval\",\n  \"interval_type\": \"<|>|<=|>=\",\n  \"value\": number,\n  \"title\": \"string\"\n}\n\n"
        "generic_explainer:\n"
        "{\n  \"mode\": \"generic_explainer\",\n  \"title\": \"string\",\n  \"sections\": [ { \"heading\": \"string\", \"bullet_points\": [ \"string\" ] } ]\n}\n\n"
        "If absolutely none fit, respond with { \"mode\": \"unsupported\", \"error\": \"unsupported\" }.\n"
        "Focus on choosing the BEST fitting schema. Return ONLY JSON."
    )


async def call_slm_with_ollama(user_prompt: str) -> Dict[str, Any]:
    """
    Calls the local Ollama instance with model 'gemma:3b', asking for STRICT JSON.
    Returns the parsed JSON as a Python dict or raises an exception.
    """
    combined_prompt = _build_system_prompt() + "\n\nUSER PROMPT:\n" + user_prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": combined_prompt,
        "stream": False,
        # Ask Ollama to return strict JSON if supported by the model
        "format": "json",
    }

    timeout = httpx.Timeout(30.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    generated_text = data.get("response")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise ValueError("Empty response from SLM")

    try:
        parsed = json.loads(generated_text)
    except Exception as exc:
        # Fallback: try to extract the first JSON object from the text
        import re
        match = re.search(r"\{[\s\S]*\}", generated_text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                raise ValueError(f"SLM did not return valid JSON: {exc}") from exc
        else:
            raise ValueError(f"SLM did not return valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("SLM JSON is not an object")

    return parsed


def _sanitize_enhanced_text(text: str) -> str:
    """Strip code fences/backticks while preserving line structure for readability."""
    import re
    # Remove triple backtick blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline backticks but keep contents
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove any accidental JSON code blocks (heuristic: lines starting with { )
    # but do not nuke braces inside normal sentences; remove whole blocks spanning braces
    text = re.sub(r"\n?\{[\s\S]*?\}\n?", "\n", text)
    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse >2 blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _fallback_structured(raw_prompt: str) -> str:
    """Produce a minimal structured template if LLM fails or echoes input."""
    rp = raw_prompt.strip()
    # Heuristic for mode detection
    lowered = rp.lower()
    vector_like = any(token in lowered for token in ["vector", "arrow", "tip-to-tail", "+", "("])
    # crude coordinate pattern check
    import re
    coord_found = re.search(r"\([-+]?\d+\s*,\s*[-+]?\d+\)", rp)
    likely_mode = "vector_addition" if vector_like and coord_found else "function_plot"
    if likely_mode == "function_plot":
        return (
            "Concept:\n"
            f"  Clarify and visualize the user's intent: {rp}.\n\n"
            "Likely mode:\n"
            "  function_plot\n\n"
            "Details:\n"
            "  - Function: f(x) = (specify based on prompt).\n"
            "  - Domain: x from (infer start) to (infer end).\n"
            "  - Approximate y-range: infer from function behavior.\n"
            "  - Highlight: note intercepts or extrema if applicable.\n\n"
            "Title:\n"
            f"  \"{rp[:60]}\""
        )
    else:
        return (
            "Concept:\n"
            f"  Clarify and visualize vector addition described by: {rp}.\n\n"
            "Likely mode:\n"
            "  vector_addition\n\n"
            "Details:\n"
            "  - Vector 1: (specify).\n"
            "  - Vector 2: (specify).\n"
            "  - Use the tip-to-tail method.\n"
            "  - Draw the resultant from origin to final tip.\n\n"
            "Title:\n"
            f"  \"Vector addition: {rp[:40]}\""
        )


async def enhance_prompt_with_llm(raw_prompt: str) -> tuple[str, str]:
    """Rewrite raw prompt into structured natural language (no JSON/code).

    Returns (enhanced_text, source) where source is 'gemini', 'openai', or 'fallback'.
    """
    original = raw_prompt.strip()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            # Resolve model name, mapping legacy aliases to current names
            configured = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            legacy_map = {
                "gemini-1.5-flash": "gemini-flash-latest",
                "gemini-1.5-pro": "gemini-pro-latest",
            }
            resolved_model = legacy_map.get(configured, configured)
            system_instructions = (
                "You rewrite casual, shorthand, or fragmentary user prompts about mathematical or vector "
                "visualizations into a CLEAR, EXPLICIT specification using ONLY this template.\n\n"
                "Template (exact headers, plain text, no JSON, no code fences):\n"
                "Concept:\n  <1–2 sentences describing what to visualize>\n\n"
                "Likely mode:\n  function_plot OR\n  vector_addition\n\n"
                "Details:\n"
                "  - Function: f(x) = ...  (for plots) OR list vectors.\n"
                "  - Domain: x from <number> to <number> OR axis extents for vectors.\n"
                "  - Approximate y-range: from <number> to <number> (if inferable for plots).\n"
                "  - Highlight: maxima, minima, intercepts, or resultant vector explanation.\n\n"
                "Title:\n  \"<short descriptive title>\"\n\n"
                "MANDATORY RULES:\n"
                "- Always fill every section; infer reasonable defaults.\n"
                "- NEVER output JSON, code, markdown fences, or backticks.\n"
                "- NEVER simply repeat the user prompt; always expand.\n"
                "- If ambiguous, state assumptions explicitly.\n"
                "- Use concise bullet lines under Details beginning with '  - '."
            )
            few_shot = (
                "Example input:\n  plot sin(x) from -2pi to 2pi and show peaks\n\n"
                "Example enhanced specification:\n"
                "Concept:\n  Visualize a 2D plot of the sine function emphasizing its periodic extrema.\n\n"
                "Likely mode:\n  function_plot\n\n"
                "Details:\n  - Function: f(x) = sin(x).\n  - Domain: x from -2π to 2π.\n  - Approximate y-range: from -1.2 to 1.2 (padding around true range).\n  - Highlight: mark each local maximum and minimum with a point and subtle label.\n\n"
                "Title:\n  \"Sine wave with extrema\"\n\n"
                "Example input:\n  vectors (2,1) and (-1,3) show their sum\n\n"
                "Example enhanced specification:\n"
                "Concept:\n  Illustrate 2D vector addition using the tip-to-tail method.\n\n"
                "Likely mode:\n  vector_addition\n\n"
                "Details:\n  - Vector 1: v1 = (2, 1).\n  - Vector 2: v2 = (-1, 3).\n  - Use the tip-to-tail method.\n  - Draw the resultant vector from origin to final tip (1, 4).\n\n"
                "Title:\n  \"Vector addition of (2,1) and (-1,3)\"\n"
                "Example input:\n  Explain sorting of array [7,4,2,1,9] using bubble sort\n\n"
                "Example enhanced specification:\n"
                "Concept:\n  Visualize the step-by-step process of Bubble Sort on the given array, showing comparisons and swaps across passes.\n\n"
                "Likely mode:\n  bubble_sort_visualization\n\n"
                "Details:\n  - Array: [7, 4, 2, 1, 9].\n  - Duration: about 10 seconds.\n  - Highlight: emphasize pairs being compared each step and show the final sorted array [1, 2, 4, 7, 9].\n\n"
                "Title:\n  \"Bubble Sort Demo for [7,4,2,1,9]\"\n"
            )
            model = genai.GenerativeModel(resolved_model, system_instruction=system_instructions)
            prompt = (
                f"{few_shot}\n\n"
                "Task: Using the same style, rewrite the following input into the required template.\n\n"
                f"Input to rewrite:\n  {original}\n\n"
                "Return ONLY the filled template (no examples, no commentary)."
            )
            resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
            enhanced = getattr(resp, "text", "") or ""
            enhanced = _sanitize_enhanced_text(enhanced)
            if not enhanced:
                return _fallback_structured(original), "fallback"
            # If model echoed or trivially short, fallback
            if enhanced.strip().lower() == original.lower() or len(enhanced.strip().split()) < 12:
                return _fallback_structured(original), "fallback"
            # Ensure required headers exist; if any missing, fallback
            required_headers = ["Concept:", "Likely mode:", "Details:", "Title:"]
            if not all(h in enhanced for h in required_headers):
                return _fallback_structured(original), "fallback"
            return enhanced, "gemini"
        except Exception:
            return _fallback_structured(original), "fallback"

    # Gemini not configured: try OpenAI or fallback directly
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            system_msg = (
                "Rewrite math/vector visualization prompts into structured natural language using the template. "
                "NO JSON, NO code, ALWAYS expand. Template sections: Concept:, Likely mode:, Details:, Title:."
            )
            user_msg = f"User prompt: {original}\nReturn only the filled template."
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.2,
            )
            enhanced = resp["choices"][0]["message"]["content"].strip()
            enhanced = _sanitize_enhanced_text(enhanced)
            if not enhanced or enhanced.lower() == original.lower():
                return _fallback_structured(original), "fallback"
            return enhanced, "openai"
        except Exception:
            return _fallback_structured(original), "fallback"

    # No LLM configured
    return _fallback_structured(original), "fallback"


def validate_slm_instructions(payload: Dict[str, Any]) -> InstructionResponse:
    mode = payload.get("mode")
    allowed_modes = {
        "function_plot",
        "vector_addition",
        "bubble_sort_visualization",
        "parametric_plot",
        "geometry_construction",
        "matrix_visualization",
        "text_step_derivation",
        "number_line_interval",
        "generic_explainer",
        "unsupported",
    }
    if mode not in allowed_modes:
        return InstructionResponse(status="error", message="Invalid JSON from SLM: unknown mode")

    if mode == "unsupported":
        return InstructionResponse(status="error", message="Unsupported prompt", mode=None, instructions=None)

    if mode == "function_plot":
        required = [
            "mode",
            "function_expression",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "duration_seconds",
            "title",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for function_plot: {', '.join(missing)}")

        try:
            # minimal type normalization
            instructions = {
                "mode": "function_plot",
                "function_expression": str(payload["function_expression"]),
                "x_min": float(payload["x_min"]),
                "x_max": float(payload["x_max"]),
                "y_min": float(payload["y_min"]),
                "y_max": float(payload["y_max"]),
                "duration_seconds": float(payload["duration_seconds"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for function_plot: {exc}")

        return InstructionResponse(status="ok", mode="function_plot", instructions=instructions)

    if mode == "vector_addition":
        required = ["mode", "vectors", "show_resultant", "show_tip_to_tail", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for vector_addition: {', '.join(missing)}")

        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) == 0:
            return InstructionResponse(status="error", message="'vectors' must be a non-empty list")

        norm_vectors: List[Dict[str, Any]] = []
        try:
            for item in vectors:
                norm_vectors.append({
                    "label": str(item["label"]),
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                })
            instructions = {
                "mode": "vector_addition",
                "vectors": norm_vectors,
                "show_resultant": bool(payload["show_resultant"]),
                "show_tip_to_tail": bool(payload["show_tip_to_tail"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for vector_addition: {exc}")

        return InstructionResponse(status="ok", mode="vector_addition", instructions=instructions)

    if mode == "bubble_sort_visualization":
        required = ["mode", "array", "duration_seconds", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for bubble_sort_visualization: {', '.join(missing)}")
        try:
            arr = [int(x) for x in payload.get("array", [])]
            instructions = {
                "mode": "bubble_sort_visualization",
                "array": arr,
                "duration_seconds": float(payload["duration_seconds"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for bubble_sort_visualization: {exc}")
        return InstructionResponse(status="ok", mode="bubble_sort_visualization", instructions=instructions)

    if mode == "parametric_plot":
        required = ["mode", "x_expression", "y_expression", "t_min", "t_max", "duration_seconds", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for parametric_plot: {', '.join(missing)}")
        try:
            instructions = {
                "mode": "parametric_plot",
                "x_expression": str(payload["x_expression"]),
                "y_expression": str(payload["y_expression"]),
                "t_min": float(payload["t_min"]),
                "t_max": float(payload["t_max"]),
                "duration_seconds": float(payload["duration_seconds"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for parametric_plot: {exc}")
        return InstructionResponse(status="ok", mode="parametric_plot", instructions=instructions)

    if mode == "geometry_construction":
        required = ["mode", "description_steps", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for geometry_construction: {', '.join(missing)}")
        steps = payload.get("description_steps")
        if not isinstance(steps, list) or len(steps) == 0:
            return InstructionResponse(status="error", message="'description_steps' must be a non-empty list")
        instructions = {
            "mode": "geometry_construction",
            "description_steps": [str(s) for s in steps],
            "title": str(payload["title"]),
        }
        return InstructionResponse(status="ok", mode="geometry_construction", instructions=instructions)

    if mode == "matrix_visualization":
        required = ["mode", "matrix", "highlight", "index", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for matrix_visualization: {', '.join(missing)}")
        mat = payload.get("matrix")
        if not isinstance(mat, list) or not mat or not all(isinstance(r, list) and r for r in mat):
            return InstructionResponse(status="error", message="'matrix' must be a non-empty 2D list")
        try:
            norm = [[float(x) for x in row] for row in mat]
            instructions = {
                "mode": "matrix_visualization",
                "matrix": norm,
                "highlight": str(payload["highlight"]),
                "index": int(payload["index"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for matrix_visualization: {exc}")
        return InstructionResponse(status="ok", mode="matrix_visualization", instructions=instructions)

    if mode == "text_step_derivation":
        required = ["mode", "steps", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for text_step_derivation: {', '.join(missing)}")
        steps = payload.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            return InstructionResponse(status="error", message="'steps' must be a non-empty list")
        instructions = {
            "mode": "text_step_derivation",
            "steps": [str(s) for s in steps],
            "title": str(payload["title"]),
        }
        return InstructionResponse(status="ok", mode="text_step_derivation", instructions=instructions)

    if mode == "number_line_interval":
        required = ["mode", "interval_type", "value", "title"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for number_line_interval: {', '.join(missing)}")
        try:
            instructions = {
                "mode": "number_line_interval",
                "interval_type": str(payload["interval_type"]),
                "value": float(payload["value"]),
                "title": str(payload["title"]),
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for number_line_interval: {exc}")
        return InstructionResponse(status="ok", mode="number_line_interval", instructions=instructions)

    if mode == "generic_explainer":
        required = ["mode", "title", "sections"]
        missing = [k for k in required if k not in payload]
        if missing:
            return InstructionResponse(status="error", message=f"Missing keys for generic_explainer: {', '.join(missing)}")
        sections = payload.get("sections")
        if not isinstance(sections, list) or len(sections) == 0:
            return InstructionResponse(status="error", message="'sections' must be a non-empty list")
        norm_sections: List[Dict[str, Any]] = []
        try:
            for sec in sections:
                heading = str(sec.get("heading"))
                bullets = sec.get("bullet_points", [])
                if not heading or not isinstance(bullets, list):
                    continue
                norm_sections.append({"heading": heading, "bullet_points": [str(b) for b in bullets]})
            instructions = {
                "mode": "generic_explainer",
                "title": str(payload["title"]),
                "sections": norm_sections,
            }
        except Exception as exc:
            return InstructionResponse(status="error", message=f"Invalid types for generic_explainer: {exc}")
        return InstructionResponse(status="ok", mode="generic_explainer", instructions=instructions)

    # Fallback (should not reach)
    return InstructionResponse(status="error", message="Invalid JSON from SLM")


@app.post("/generate/instructions", response_model=InstructionResponse)
async def generate_instructions(req: InstructionRequest) -> InstructionResponse:
    try:
        # 1) Enhance the prompt using cloud LLM (plain English only)
        enhanced, enhanced_source = await enhance_prompt_with_llm(req.prompt)
        # 2) Pass enhanced text to SLM (Ollama/Gemma) to get STRICT JSON
        slm_json = await call_slm_with_ollama(enhanced)
    except httpx.HTTPError as exc:
        # Network/HTTP issues talking to Ollama
        return InstructionResponse(status="error", message=f"SLM HTTP error: {exc}", enhanced_prompt=locals().get("enhanced"), enhanced_source=locals().get("enhanced_source"))
    except Exception as exc:
        return InstructionResponse(status="error", message=str(exc), enhanced_prompt=locals().get("enhanced"), enhanced_source=locals().get("enhanced_source"))

    # Validate and normalize into known schema
    validated = validate_slm_instructions(slm_json)
    # Attach enhanced prompt for UI/debugging
    validated.enhanced_prompt = locals().get("enhanced")
    validated.enhanced_source = locals().get("enhanced_source")
    return validated


# Development hint:
# Run the API with:
#   uvicorn app:app --reload
# Then test:
#   curl http://127.0.0.1:8000/health
#   curl -X POST http://127.0.0.1:8000/render/function_plot \
#        -H "Content-Type: application/json" \
#        -d '{
#              "mode": "function_plot",
#              "function_expression": "sin(x)",
#              "x_min": -6.28,
#              "x_max": 6.28,
#              "y_min": -2.0,
#              "y_max": 2.0,
#              "duration_seconds": 8,
#              "title": "Sine Wave with Maxima and Minima"
#            }'
