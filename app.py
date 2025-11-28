from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Dict
import os
import json
import re

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Registry-based rendering
from animation_modes import render_animation_from_mode, MODE_REGISTRY

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
    """Report LLM enhancer configuration without exposing secrets."""
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

    # Package availability
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


# =============================================================================
# Generic Rendering Endpoint
# =============================================================================

class AnyModeRequest(BaseModel):
    mode: str
    payload: Dict[str, Any]


async def retry_fix_manim_code(original_code: str, error_trace: str) -> Optional[str]:
    """Ask SLM to fix the broken Manim code."""
    # Construct a clear instruction prompt
    prompt = (
        "### INSTRUCTION ###\n"
        "The following Manim code has an error.\n"
        "CODE:\n"
        f"```python\n{original_code}\n```\n"
        "ERROR:\n"
        f"{error_trace}\n\n"
        "TASK: Fix the code. Output ONLY the valid JSON with the fixed code.\n"
        "JSON format: {\"mode\": \"manim_code_gen\", \"code\": \"...\"}\n"
    )
    
    try:
        # Use manim_code_gen mode to get JSON wrapper with "code" field
        response = await call_slm_with_ollama(prompt, target_mode="manim_code_gen")
        code = response.get("code", "")
        
        # Validation: Remove markdown code blocks if present inside the string
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Validation: Ensure it doesn't just echo the prompt
        if "### INSTRUCTION ###" in code or "You previously generated" in code:
            print("DEBUG: SLM echoed prompt. Discarding.")
            return None
            
        # Validation: Ensure it looks like code
        if not ("class GeneratedScene" in code and "construct" in code):
            print("DEBUG: Fixed code missing required structure.")
            return None
            
        return code
    except Exception as e:
        print(f"Fix attempt failed: {e}")
        return None

@app.post("/render/any_mode", response_model=RenderResponse)
async def render_any_mode_endpoint(req: AnyModeRequest) -> RenderResponse:
    """Universal endpoint to render any supported animation mode."""
    try:
        # The payload might contain the mode inside it, or not. 
        # We ensure it's passed to the registry helper.
        data = req.payload.copy()
        # If mode is missing in payload, inject it from the request wrapper
        if "mode" not in data:
            data["mode"] = req.mode
            
        out_path = await run_in_threadpool(render_animation_from_mode, req.mode, data)
    except Exception as exc:
        # Auto-retry for Manim code generation errors
        if req.mode == "manim_code_gen" and "Manim execution failed" in str(exc):
             print("DEBUG: Manim failed. Attempting auto-fix...")
             original_code = data.get("code", "")
             error_trace = str(exc)
             
             fixed_code = await retry_fix_manim_code(original_code, error_trace)
             
             if fixed_code:
                 print("DEBUG: Applying fixed code and retrying...")
                 data["code"] = fixed_code
                 try:
                     out_path = await run_in_threadpool(render_animation_from_mode, req.mode, data)
                     return RenderResponse(status="ok", output_video_path=str(out_path))
                 except Exception as retry_exc:
                     print(f"DEBUG: Retry failed: {retry_exc}")
                     raise HTTPException(status_code=500, detail=f"Rendering failed after retry: {retry_exc}")
        
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc))
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


# =============================================================================
# Legacy/Specific Endpoints (Refactored to use Registry)
# =============================================================================

@app.post("/render/function_plot", response_model=RenderResponse)
def render_function_plot_endpoint(req: Dict[str, Any]) -> RenderResponse:
    """Legacy endpoint for function_plot."""
    # We accept Dict to avoid double validation, or we could use the Pydantic model.
    # Using Dict allows us to just pass it through.
    # But to keep API docs clean, we should ideally use the Pydantic model.
    # For now, let's just use the registry.
    
    # Note: req is a dict if we don't type hint it as a Pydantic model in the signature.
    # But FastAPI wants a model for docs. 
    # Let's use the registry's config model if available, or just a generic dict wrapper.
    # To preserve backward compatibility with the frontend's strict typing, we'll just
    # forward the request to the generic handler logic.
    
    if req.get("mode") != "function_plot":
         raise HTTPException(status_code=400, detail="Unsupported mode; expected 'function_plot'.")

    try:
        out_path = render_animation_from_mode("function_plot", req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


@app.post("/render/vector_addition", response_model=RenderResponse)
def render_vector_addition_endpoint(req: Dict[str, Any]) -> RenderResponse:
    if req.get("mode") != "vector_addition":
        raise HTTPException(status_code=400, detail="Unsupported mode; expected 'vector_addition'.")
    try:
        out_path = render_animation_from_mode("vector_addition", req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


@app.post("/render/bubble_sort_visualization", response_model=RenderResponse)
def render_bubble_sort_endpoint(req: Dict[str, Any]) -> RenderResponse:
    if req.get("mode") != "bubble_sort_visualization":
        raise HTTPException(status_code=400, detail="Unsupported mode; expected 'bubble_sort_visualization'.")
    try:
        out_path = render_animation_from_mode("bubble_sort_visualization", req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {exc}")
    return RenderResponse(status="ok", output_video_path=str(out_path))


# ================================
# SLM (Ollama) integration section
# ================================

# SLM NOTE:
# We use `qwen3:4b` via Ollama as our local Small Language Model for JSON generation.
# The pipeline remains:
#   Gemini (cloud LLM) → Qwen3-4B (SLM) → JSON → Manim.
# Swapping SLM models only requires changing SLM_MODEL and adjusting this system prompt.

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"
MANIM_MODEL = "qwen2.5-coder:7b"

# Load Dataset for RAG (In-Context Learning)
DATASET = []
try:
    with open("dataset.json", "r") as f:
        DATASET = json.load(f)
    print(f"Loaded {len(DATASET)} examples from dataset.json")
except Exception as e:
    print(f"Warning: Could not load dataset.json: {e}")

def get_few_shot_examples(target_mode: str, limit: int = 3) -> str:
    """Retrieve relevant examples from the dataset for the given mode."""
    examples = [ex for ex in DATASET if ex["response"].get("mode") == target_mode]
    
    if not examples:
        return ""

    # Randomly select examples to provide variety
    import random
    selected = random.sample(examples, min(limit, len(examples)))
    
    output = "\n\n### REFERENCE EXAMPLES (LEARN FROM THESE):\n"
    for i, ex in enumerate(selected):
        output += f"\n--- Example {i+1} ---\n"
        output += f"User Prompt: {ex['prompt']}\n"
        
        if target_mode == "manim_code_gen":
             code = ex["response"].get("code", "")
             output += f"Correct Code:\n```python\n{code}\n```\n"
        else:
             output += f"Correct JSON Response: {json.dumps(ex['response'])}\n"
    
    output += "\n### END EXAMPLES\n"
    return output


class InstructionRequest(BaseModel):
    prompt: str


class InstructionResponse(BaseModel):
    status: str  # "ok" or "error"
    mode: Optional[str] = None
    instructions: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    enhanced_source: Optional[str] = None


def _build_system_prompt() -> str:
    """
    Builds the system prompt for the SLM.
    CRITICAL: Enforces strict adherence to TARGET_MODE from the enhanced prompt.
    """
    return (
        "You are a SMALL, CODE-ORIENTED LANGUAGE MODEL.\n\n"
        "Your ONLY job:\n"
        "Given an enhanced prompt with fields:\n\n"
        "TARGET_MODE: <mode>\n"
        "HIGH_LEVEL_INSTRUCTIONS: <text>\n"
        "KEY_VALUES: <key=value; key=value; ...>\n"
        "REQUIREMENTS: <text>\n\n"
        "you must generate EXACTLY ONE JSON object that describes a Manim animation for that mode.\n\n"
        "Supported modes and their JSON schemas:\n\n"
        "1) function_plot\n"
        "----------------\n"
        "{\n  \"mode\": \"function_plot\",\n  \"function_expression\": \"sin(x)\",\n  \"x_min\": -6.28,\n  \"x_max\": 6.28,\n  \"y_min\": -2,\n  \"y_max\": 2,\n  \"duration_seconds\": 8,\n  \"title\": \"Sine Wave with Extrema\"\n}\n\n"
        "2) parametric_plot\n"
        "------------------\n"
        "{\n  \"mode\": \"parametric_plot\",\n  \"x_expression\": \"cos(t)\",\n  \"y_expression\": \"sin(t)\",\n  \"t_min\": 0,\n  \"t_max\": 6.28,\n  \"duration_seconds\": 8,\n  \"title\": \"Unit Circle\"\n}\n\n"
        "3) vector_addition\n"
        "------------------\n"
        "{\n  \"mode\": \"vector_addition\",\n  \"vectors\": [\n    { \"label\": \"v1\", \"x\": 2, \"y\": 1 },\n    { \"label\": \"v2\", \"x\": -1, \"y\": 3 }\n  ],\n  \"show_resultant\": true,\n  \"show_tip_to_tail\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Vector Addition Demo\"\n}\n\n"
        "4) bubble_sort_visualization\n"
        "----------------------------\n"
        "{\n  \"mode\": \"bubble_sort_visualization\",\n  \"array\": [5, 1, 4, 2],\n  \"duration_seconds\": 10,\n  \"title\": \"Bubble Sort Demo\"\n}\n\n"
        "5) geometry_construction\n"
        "------------------------\n"
        "{\n  \"mode\": \"geometry_construction\",\n  \"description_steps\": [\n    \"Draw triangle ABC.\",\n    \"Construct the perpendicular bisector of AB.\"\n  ],\n  \"duration_seconds\": 10,\n  \"title\": \"Perpendicular Bisector Construction\"\n}\n\n"
        "6) matrix_visualization\n"
        "-----------------------\n"
        "{\n  \"mode\": \"matrix_visualization\",\n  \"matrix_A\": [[1, 2], [3, 4]],\n  \"matrix_B\": [[5, 6], [7, 8]],\n  \"operation\": \"multiplication\",\n  \"result_matrix\": [[19, 22], [43, 50]],\n  \"highlight_color_A\": \"BLUE\",\n  \"highlight_color_B\": \"GREEN\",\n  \"show_intermediate_steps\": true,\n  \"duration_seconds\": 12,\n  \"title\": \"Matrix Multiplication (2x2)\"\n}\n\n"
        "7) number_line_interval\n"
        "-----------------------\n"
        "{\n  \"mode\": \"number_line_interval\",\n  \"left_value\": -5,\n  \"right_value\": 7,\n  \"include_left\": false,\n  \"include_right\": true,\n  \"interval_type\": \"<\",\n  \"duration_seconds\": 8,\n  \"title\": \"-5 < x ≤ 7\"\n}\n\n"
        "8) text_step_derivation\n"
        "-----------------------\n"
        "{\n  \"mode\": \"text_step_derivation\",\n  \"steps\": [\n    \"Start with the quadratic equation ax^2 + bx + c = 0.\",\n    \"Divide both sides by a.\",\n    \"Complete the square on the left-hand side.\"\n  ],\n  \"duration_seconds\": 12,\n  \"title\": \"Quadratic Formula Derivation\"\n}\n\n"
        "9) generic_explainer\n"
        "--------------------\n"
        "{\n  \"mode\": \"generic_explainer\",\n  \"title\": \"Photosynthesis Overview\",\n  \"sections\": [\n    {\n      \"heading\": \"What is Photosynthesis?\",\n      \"bullet_points\": [\n        \"Plants convert light energy into chemical energy.\",\n        \"The process occurs mainly in the leaves.\"\n      ]\n    },\
    {\n      \"heading\": \"Inputs and Outputs\",\n      \"bullet_points\": [\n        \"Inputs: carbon dioxide, water, and sunlight.\",\n        \"Outputs: glucose and oxygen.\"\n      ]\n    }\n  ],\n  \"duration_seconds\": 10\n}\n\n"
        "10) derivative_visualization\n"
        "----------------------------\n"
        "{\n  \"mode\": \"derivative_visualization\",\n  \"function_expression\": \"x**2\",\n  \"x_min\": -3,\n  \"x_max\": 3,\n  \"x0\": 1,\n  \"show_derivative_curve\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Derivative of x^2 at x=1\"\n}\n\n"
        "11) integral_area_visualization\n"
        "-------------------------------\n"
        "{\n  \"mode\": \"integral_area_visualization\",\n  \"function_expression\": \"sin(x)\",\n  \"a\": 0,\n  \"b\": 3.14,\n  \"num_rectangles\": 20,\n  \"method\": \"midpoint\",\n  \"show_exact_area_label\": true,\n  \"duration_seconds\": 10,\n  \"title\": \"Area under sin(x)\"\n}\n\n"
        "12) limit_visualization\n"
        "-----------------------\n"
        "{\n  \"mode\": \"limit_visualization\",\n  \"function_expression\": \"(x**2 - 1)/(x - 1)\",\n  \"x_min\": -1,\n  \"x_max\": 3,\n  \"x0\": 1,\n  \"show_left_right\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Limit as x->1\"\n}\n\n"
        "13) manim_code_gen\n"
        "------------------\n"
        "{\n  \"mode\": \"manim_code_gen\",\n  \"code\": \"from manim import *\\n\\nclass GeneratedScene(Scene):\\n    def construct(self):\\n        c = Circle()\\n        self.play(Create(c))\",\n  \"title\": \"Custom Animation\"\n}\n\n"
        "IMPORTANT RULES:\n\n"
        "- Read TARGET_MODE. If it is one of:\n"
        "  function_plot, parametric_plot, vector_addition, bubble_sort_visualization,\n"
        "  geometry_construction, matrix_visualization, text_step_derivation, number_line_interval, generic_explainer,\n"
        "  derivative_visualization, integral_area_visualization, limit_visualization, manim_code_gen\n\n"
        "  → You MUST use exactly that mode in the \"mode\" field.\n"
        "  → Do NOT change it to another mode.\n"
        "  → Do NOT default to generic_explainer if TARGET_MODE is valid.\n\n"
        "- Use HIGH_LEVEL_INSTRUCTIONS, KEY_VALUES, and REQUIREMENTS to fill fields.\n"
        "  - Parse key=value pairs from KEY_VALUES.\n"
        "  - If some values are missing, choose simple defaults (e.g., duration_seconds=8).\n"
        "  - Clamp array sizes to at most 10 elements for sorting.\n"
        "  - Clamp bullet_points to at most 4 per section in generic_explainer.\n\n"
        "- NEVER output error messages like:\n"
        "  \"Could not parse details\" or \"Please try a more specific prompt\".\n"
        "  If something is unclear, make a reasonable guess and still output valid JSON.\n\n"
        "- Your response MUST be:\n"
        "  - A single JSON object\n"
        "  - No backticks, no markdown, no comments, no extra text.\n\n"
        "FEW-SHOT EXAMPLES:\n\n"
        "Example 1 (INEQUALITY):\n\n"
        "INPUT:\n"
        "TARGET_MODE: number_line_interval\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show the inequality -5 < x ≤ 7 on a number line.\n"
        "KEY_VALUES: inequality_string=-5 < x <= 7; left_value=-5; right_value=7; include_left=false; include_right=true\n"
        "REQUIREMENTS: duration_seconds=8; label_endpoints=true; shade_interval=true\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"number_line_interval\",\n"
        "  \"left_value\": -5,\n"
        "  \"right_value\": 7,\n"
        "  \"include_left\": false,\n"
        "  \"include_right\": true,\n"
        "  \"interval_type\": \"<\",\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"-5 < x ≤ 7\"\n"
        "}\n\n"
        "Example 2 (MATRIX):\n\n"
        "INPUT:\n"
        "TARGET_MODE: matrix_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show matrices A and B and animate their product C.\n"
        "KEY_VALUES: matrix_A=[[1,2],[3,4]]; matrix_B=[[5,6],[7,8]]; operation=multiplication; result_matrix=[[19,22],[43,50]]\n"
        "REQUIREMENTS: duration_seconds=15; title=\"Matrix Multiplication (2x2)\"; highlight_color_A=BLUE; highlight_color_B=GREEN; show_intermediate_steps=True.\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"matrix_visualization\",\n"
        "  \"matrix_A\": [[1, 2], [3, 4]],\n"
        "  \"matrix_B\": [[5, 6], [7, 8]],\n"
        "  \"operation\": \"multiplication\",\n"
        "  \"result_matrix\": [[19, 22], [43, 50]],\n"
        "  \"highlight_color_A\": \"BLUE\",\n"
        "  \"highlight_color_B\": \"GREEN\",\n"
        "  \"show_intermediate_steps\": true,\n"
        "  \"duration_seconds\": 15,\n"
        "  \"title\": \"Matrix Multiplication (2x2)\"\n"
        "}\n\n"
        "Example 3 (GENERIC EXPLAINER):\n\n"
        "INPUT:\n"
        "TARGET_MODE: generic_explainer\n"
        "HIGH_LEVEL_INSTRUCTIONS: Explain photosynthesis simply.\n"
        "KEY_VALUES: topic=photosynthesis\n"
        "REQUIREMENTS: duration_seconds=10; sections=3\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"generic_explainer\",\n"
        "  \"title\": \"Photosynthesis Overview\",\n"
        "  \"sections\": [\n"
        "    {\n"
        "      \"heading\": \"What is Photosynthesis?\",\n"
        "      \"bullet_points\": [\n"
        "        \"Plants convert light energy into chemical energy.\",\n"
        "        \"The process occurs mainly in the leaves.\"\n"
        "      ]\n"
        "    },\
    {\n      \"heading\": \"Inputs and Outputs\",\n      \"bullet_points\": [\n        \"Inputs: carbon dioxide, water, and sunlight.\",\n        \"Outputs: glucose and oxygen.\"\n      ]\n    }\n  ],\n  \"duration_seconds\": 10\n}\n\n"
        "Example 4 (DERIVATIVE):\n\n"
        "INPUT:\n"
        "TARGET_MODE: derivative_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show derivative of f(x) = x^2 at x0 = 1 with tangent line and derivative curve.\n"
        "KEY_VALUES: function_expression=x**2; x_min=-3; x_max=3; x0=1\n"
        "REQUIREMENTS: show_derivative_curve=true; duration_seconds=8\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"derivative_visualization\",\n"
        "  \"function_expression\": \"x**2\",\n"
        "  \"x_min\": -3,\n"
        "  \"x_max\": 3,\n"
        "  \"x0\": 1,\n"
        "  \"show_derivative_curve\": true,\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"Derivative of x^2 at x=1\"\n"
        "}\n\n"
        "Example 5 (INTEGRAL):\n\n"
        "INPUT:\n"
        "TARGET_MODE: integral_area_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Visualize the area under sin(x) from 0 to π using Riemann rectangles.\n"
        "KEY_VALUES: function_expression=sin(x); a=0; b=3.14; num_rectangles=20; method=midpoint\n"
        "REQUIREMENTS: shade region and show approximate area.\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"integral_area_visualization\",\n"
        "  \"function_expression\": \"sin(x)\",\n"
        "  \"a\": 0,\n"
        "  \"b\": 3.14,\n"
        "  \"num_rectangles\": 20,\n"
        "  \"method\": \"midpoint\",\n"
        "  \"show_exact_area_label\": false,\n"
        "  \"duration_seconds\": 10,\n"
        "  \"title\": \"Area under sin(x) from 0 to π\"\n"
        "}\n\n"
        "Example 6 (LIMIT):\n\n"
        "INPUT:\n"
        "TARGET_MODE: limit_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show the limit of (x^2 - 1) / (x - 1) as x approaches 1.\n"
        "KEY_VALUES: function_expression=(x**2 - 1) / (x - 1); x_min=-1; x_max=3; x0=1\n"
        "REQUIREMENTS: show approach from both sides.\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"limit_visualization\",\n"
        "  \"function_expression\": \"(x**2 - 1) / (x - 1)\",\n"
        "  \"x_min\": -1,\n"
        "  \"x_max\": 3,\n"
        "  \"x0\": 1,\n"
        "  \"show_left_right\": true,\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"Limit as x → 1\"\n"
        "}\n\n"
        "Example 7 (CUSTOM PHYSICS):\n\n"
        "INPUT:\n"
        "TARGET_MODE: manim_code_gen\n"
        "HIGH_LEVEL_INSTRUCTIONS: Animate a simple circle creation.\n"
        "KEY_VALUES: object=circle\n"
        "REQUIREMENTS: basic animation\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"manim_code_gen\",\n"
        "  \"code\": \"from manim import *\\n\\nclass GeneratedScene(Scene):\\n    def construct(self):\\n        circle = Circle(radius=1, color=BLUE)\\n        self.play(Create(circle))\",\n"
        "  \"title\": \"Simple Circle\"\n"
        "}\n\n"
        "Now, for ANY enhanced prompt I send, respond ONLY with the JSON object for the given TARGET_MODE."
    )


def _build_system_prompt() -> str:
    # This function is assumed to exist and return the large string above.
    # For the purpose of this edit, we'll define it returning the content provided.
    return (
        "You are a SMALL, CODE-ORIENTED LANGUAGE MODEL.\n\n"
        "Your ONLY job:\n"
        "Given an enhanced prompt with fields:\n\n"
        "TARGET_MODE: <mode>\n"
        "HIGH_LEVEL_INSTRUCTIONS: <text>\n"
        "KEY_VALUES: <key=value; key=value; ...>\n"
        "REQUIREMENTS: <text>\n\n"
        "you must generate EXACTLY ONE JSON object that describes a Manim animation for that mode.\n\n"
        "Supported modes and their JSON schemas:\n\n"
        "1) function_plot\n"
        "----------------\n"
        "{\n  \"mode\": \"function_plot\",\n  \"function_expression\": \"sin(x)\",\n  \"x_min\": -6.28,\n  \"x_max\": 6.28,\n  \"y_min\": -2,\n  \"y_max\": 2,\n  \"duration_seconds\": 8,\n  \"title\": \"Sine Wave with Extrema\"\n}\n\n"
        "2) parametric_plot\n"
        "------------------\n"
        "{\n  \"mode\": \"parametric_plot\",\n  \"x_expression\": \"cos(t)\",\n  \"y_expression\": \"sin(t)\",\n  \"t_min\": 0,\n  \"t_max\": 6.28,\n  \"duration_seconds\": 8,\n  \"title\": \"Unit Circle\"\n}\n\n"
        "3) vector_addition\n"
        "------------------\n"
        "{\n  \"mode\": \"vector_addition\",\n  \"vectors\": [\n    { \"label\": \"v1\", \"x\": 2, \"y\": 1 },\n    { \"label\": \"v2\", \"x\": -1, \"y\": 3 }\n  ],\n  \"show_resultant\": true,\n  \"show_tip_to_tail\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Vector Addition Demo\"\n}\n\n"
        "4) bubble_sort_visualization\n"
        "----------------------------\n"
        "{\n  \"mode\": \"bubble_sort_visualization\",\n  \"array\": [5, 1, 4, 2],\n  \"duration_seconds\": 10,\n  \"title\": \"Bubble Sort Demo\"\n}\n\n"
        "5) geometry_construction\n"
        "------------------------\n"
        "{\n  \"mode\": \"geometry_construction\",\n  \"description_steps\": [\n    \"Draw triangle ABC.\",\n    \"Construct the perpendicular bisector of AB.\"\n  ],\n  \"duration_seconds\": 10,\n  \"title\": \"Perpendicular Bisector Construction\"\n}\n\n"
        "6) matrix_visualization\n"
        "-----------------------\n"
        "{\n  \"mode\": \"matrix_visualization\",\n  \"matrix_A\": [[1, 2], [3, 4]],\n  \"matrix_B\": [[5, 6], [7, 8]],\n  \"operation\": \"multiplication\",\n  \"result_matrix\": [[19, 22], [43, 50]],\n  \"highlight_color_A\": \"BLUE\",\n  \"highlight_color_B\": \"GREEN\",\n  \"show_intermediate_steps\": true,\n  \"duration_seconds\": 12,\n  \"title\": \"Matrix Multiplication (2x2)\"\n}\n\n"
        "7) number_line_interval\n"
        "-----------------------\n"
        "{\n  \"mode\": \"number_line_interval\",\n  \"left_value\": -5,\n  \"right_value\": 7,\n  \"include_left\": false,\n  \"include_right\": true,\n  \"interval_type\": \"<\",\n  \"duration_seconds\": 8,\n  \"title\": \"-5 < x ≤ 7\"\n}\n\n"
        "8) text_step_derivation\n"
        "-----------------------\n"
        "{\n  \"mode\": \"text_step_derivation\",\n  \"steps\": [\n    \"Start with the quadratic equation ax^2 + bx + c = 0.\",\n    \"Divide both sides by a.\",\n    \"Complete the square on the left-hand side.\"\n  ],\n  \"duration_seconds\": 12,\n  \"title\": \"Quadratic Formula Derivation\"\n}\n\n"
        "9) generic_explainer\n"
        "--------------------\n"
        "{\n  \"mode\": \"generic_explainer\",\n  \"title\": \"Photosynthesis Overview\",\n  \"sections\": [\n    {\n      \"heading\": \"What is Photosynthesis?\",\n      \"bullet_points\": [\n        \"Plants convert light energy into chemical energy.\",\n        \"The process occurs mainly in the leaves.\"\n      ]\n    },\n    {\n      \"heading\": \"Inputs and Outputs\",\n      \"bullet_points\": [\n        \"Inputs: carbon dioxide, water, and sunlight.\",\n        \"Outputs: glucose and oxygen.\"\n      ]\n    }\n  ],\n  \"duration_seconds\": 10\n}\n\n"
        "10) derivative_visualization\n"
        "----------------------------\n"
        "{\n  \"mode\": \"derivative_visualization\",\n  \"function_expression\": \"x**2\",\n  \"x_min\": -3,\n  \"x_max\": 3,\n  \"x0\": 1,\n  \"show_derivative_curve\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Derivative of x^2 at x=1\"\n}\n\n"
        "11) integral_area_visualization\n"
        "-------------------------------\n"
        "{\n  \"mode\": \"integral_area_visualization\",\n  \"function_expression\": \"sin(x)\",\n  \"a\": 0,\n  \"b\": 3.14,\n  \"num_rectangles\": 20,\n  \"method\": \"midpoint\",\n  \"show_exact_area_label\": true,\n  \"duration_seconds\": 10,\n  \"title\": \"Area under sin(x)\"\n}\n\n"
        "12) limit_visualization\n"
        "-----------------------\n"
        "{\n  \"mode\": \"limit_visualization\",\n  \"function_expression\": \"(x**2 - 1)/(x - 1)\",\n  \"x_min\": -1,\n  \"x_max\": 3,\n  \"x0\": 1,\n  \"show_left_right\": true,\n  \"duration_seconds\": 8,\n  \"title\": \"Limit as x->1\"\n}\n\n"
        "13) manim_code_gen\n"
        "------------------\n"
        "{\n  \"mode\": \"manim_code_gen\",\n  \"code\": \"from manim import *\\n\\nclass GeneratedScene(Scene):\\n    def construct(self):\\n        c = Circle()\\n        self.play(Create(c))\",\n  \"title\": \"Custom Animation\"\n}\n\n"
        "IMPORTANT RULES:\n\n"
        "- Read TARGET_MODE. If it is one of:\n"
        "  function_plot, parametric_plot, vector_addition, bubble_sort_visualization,\n"
        "  geometry_construction, matrix_visualization, text_step_derivation, number_line_interval, generic_explainer,\n"
        "  derivative_visualization, integral_area_visualization, limit_visualization, manim_code_gen\n\n"
        "  → You MUST use exactly that mode in the \"mode\" field.\n"
        "  → Do NOT change it to another mode.\n"
        "  → Do NOT default to generic_explainer if TARGET_MODE is valid.\n\n"
        "- Use HIGH_LEVEL_INSTRUCTIONS, KEY_VALUES, and REQUIREMENTS to fill fields.\n"
        "  - Parse key=value pairs from KEY_VALUES.\n"
        "  - If some values are missing, choose simple defaults (e.g., duration_seconds=8).\n"
        "  - Clamp array sizes to at most 10 elements for sorting.\n"
        "  - Clamp bullet_points to at most 4 per section in generic_explainer.\n\n"
        "- NEVER output error messages like:\n"
        "  \"Could not parse details\" or \"Please try a more specific prompt\".\n"
        "  If something is unclear, make a reasonable guess and still output valid JSON.\n\n"
        "- Your response MUST be:\n"
        "  - A single JSON object\n"
        "  - No backticks, no markdown, no comments, no extra text.\n\n"
        "FEW-SHOT EXAMPLES:\n\n"
        "Example 1 (INEQUALITY):\n\n"
        "INPUT:\n"
        "TARGET_MODE: number_line_interval\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show the inequality -5 < x ≤ 7 on a number line.\n"
        "KEY_VALUES: inequality_string=-5 < x <= 7; left_value=-5; right_value=7; include_left=false; include_right=true\n"
        "REQUIREMENTS: duration_seconds=8; label_endpoints=true; shade_interval=true\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"number_line_interval\",\n"
        "  \"left_value\": -5,\n"
        "  \"right_value\": 7,\n"
        "  \"include_left\": false,\n"
        "  \"include_right\": true,\n"
        "  \"interval_type\": \"<\",\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"-5 < x ≤ 7\"\n"
        "}\n\n"
        "Example 2 (MATRIX):\n\n"
        "INPUT:\n"
        "TARGET_MODE: matrix_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show matrices A and B and animate their product C.\n"
        "KEY_VALUES: matrix_A=[[1,2],[3,4]]; matrix_B=[[5,6],[7,8]]; operation=multiplication; result_matrix=[[19,22],[43,50]]\n"
        "REQUIREMENTS: duration_seconds=15; title=\"Matrix Multiplication (2x2)\"; highlight_rows=true; highlight_columns=true; show_intermediate_steps=true\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"matrix_visualization\",\n"
        "  \"matrix_A\": [[1, 2], [3, 4]],\n"
        "  \"matrix_B\": [[5, 6], [7, 8]],\n"
        "  \"operation\": \"multiplication\",\n"
        "  \"result_matrix\": [[19, 22], [43, 50]],\n"
        "  \"highlight_color_A\": \"BLUE\",\n"
        "  \"highlight_color_B\": \"GREEN\",\n"
        "  \"show_intermediate_steps\": true,\n"
        "  \"duration_seconds\": 15,\n"
        "  \"title\": \"Matrix Multiplication (2x2)\"\n"
        "}\n\n"
        "Example 3 (GENERIC EXPLAINER):\n\n"
        "INPUT:\n"
        "TARGET_MODE: generic_explainer\n"
        "HIGH_LEVEL_INSTRUCTIONS: Explain photosynthesis simply.\n"
        "KEY_VALUES: topic=photosynthesis\n"
        "REQUIREMENTS: duration_seconds=10; sections=3\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"generic_explainer\",\n"
        "  \"title\": \"Photosynthesis Overview\",\n"
        "  \"sections\": [\n"
        "    {\n"
        "      \"heading\": \"What is Photosynthesis?\",\n"
        "      \"bullet_points\": [\n"
        "        \"Plants convert light energy into chemical energy.\",\n"
        "        \"The process occurs mainly in the leaves.\"\n"
        "      ]\n"
        "    },\n"
        "    {\n"
        "      \"heading\": \"Inputs and Outputs\",\n"
        "      \"bullet_points\": [\n"
        "        \"Inputs: carbon dioxide, water, and sunlight.\",\n"
        "        \"Outputs: glucose and oxygen.\"\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        "  \"duration_seconds\": 10\n"
        "}\n\n"
        "Example 4 (DERIVATIVE):\n\n"
        "INPUT:\n"
        "TARGET_MODE: derivative_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show derivative of f(x) = x^2 at x0 = 1 with tangent line and derivative curve.\n"
        "KEY_VALUES: function_expression=x**2; x_min=-3; x_max=3; x0=1\n"
        "REQUIREMENTS: show_derivative_curve=true; duration_seconds=8\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"derivative_visualization\",\n"
        "  \"function_expression\": \"x**2\",\n"
        "  \"x_min\": -3,\n"
        "  \"x_max\": 3,\n"
        "  \"x0\": 1,\n"
        "  \"show_derivative_curve\": true,\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"Derivative of x^2 at x=1\"\n"
        "}\n\n"
        "Example 5 (INTEGRAL):\n\n"
        "INPUT:\n"
        "TARGET_MODE: integral_area_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Visualize the area under sin(x) from 0 to π using Riemann rectangles.\n"
        "KEY_VALUES: function_expression=sin(x); a=0; b=3.14; num_rectangles=20; method=midpoint\n"
        "REQUIREMENTS: shade region and show approximate area.\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"integral_area_visualization\",\n"
        "  \"function_expression\": \"sin(x)\",\n"
        "  \"a\": 0,\n"
        "  \"b\": 3.14,\n"
        "  \"num_rectangles\": 20,\n"
        "  \"method\": \"midpoint\",\n"
        "  \"show_exact_area_label\": false,\n"
        "  \"duration_seconds\": 10,\n"
        "  \"title\": \"Area under sin(x) from 0 to π\"\n"
        "}\n\n"
        "Example 6 (LIMIT):\n\n"
        "INPUT:\n"
        "TARGET_MODE: limit_visualization\n"
        "HIGH_LEVEL_INSTRUCTIONS: Show the limit of (x^2 - 1) / (x - 1) as x approaches 1.\n"
        "KEY_VALUES: function_expression=(x**2 - 1) / (x - 1); x_min=-1; x_max=3; x0=1\n"
        "REQUIREMENTS: show approach from both sides.\n\n"
        "OUTPUT:\n"
        "{\n"
        "  \"mode\": \"limit_visualization\",\n"
        "  \"function_expression\": \"(x**2 - 1) / (x - 1)\",\n"
        "  \"x_min\": -1,\n"
        "  \"x_max\": 3,\n"
        "  \"x0\": 1,\n"
        "  \"show_left_right\": true,\n"
        "  \"duration_seconds\": 8,\n"
        "  \"title\": \"Limit as x → 1\"\n"
        "}\n\n"
        "Now, for ANY enhanced prompt I send, respond ONLY with the JSON object for the given TARGET_MODE."
    )


def _build_code_gen_system_prompt() -> str:
    return (
        "You are a Python expert specializing in Manim animations.\n"
        "Your task is to write a complete, runnable Manim script based on the user's request.\n"
        "RULES:\n"
        "1. Output ONLY valid Python code.\n"
        "2. The script MUST define a class named 'GeneratedScene' inheriting from 'Scene'.\n"
        "3. Import everything needed: 'from manim import *'.\n"
        "4. Do NOT use Markdown backticks (```python). Just raw code.\n"
        "5. Do NOT add explanations or text outside the code.\n"
    )

async def call_slm_with_ollama(user_prompt: str, target_mode: str = "json") -> Dict[str, Any]:
    """
    Call the local SLM via Ollama to generate JSON instructions or raw code.
    """
    # Choose model
    model_name = OLLAMA_MODEL
    if target_mode == "manim_code_gen":
        model_name = MANIM_MODEL
        system_prompt = _build_code_gen_system_prompt()
    else:
        system_prompt = _build_system_prompt()
    
    # Construct payload
    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low temp for deterministic output
            "num_predict": 4096, # More tokens for code
            "repeat_penalty": 1.1, # Standard penalty for 7B
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_response = data.get("response", "")
            
            # If code gen mode, just return the raw response wrapped
            if target_mode == "manim_code_gen":
                # Combine prompt + response for full code (since we used completion prompt)
                full_code = user_prompt + raw_response
                
                # Clean up markdown backticks if present (unlikely with completion, but safe)
                clean_code = full_code
                code_match = re.search(r"```python\s*([\s\S]*?)```", full_code)
                if not code_match:
                    code_match = re.search(r"```\s*([\s\S]*?)```", full_code)
                
                if code_match:
                    clean_code = code_match.group(1)
                
                # Fix truncated code (basic)
                open_p = clean_code.count('(')
                close_p = clean_code.count(')')
                if open_p > close_p:
                    clean_code += ')' * (open_p - close_p)
                
                # Ensure it ends with newline
                clean_code += "\n"

                return {
                    "mode": "manim_code_gen",
                    "code": clean_code,
                    "_raw_response": raw_response
                }
            
            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", raw_response)
            parsed = None
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # If regex failed or parse failed, try whole string
            if parsed is None:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    pass
            
            if parsed is not None and isinstance(parsed, dict):
                parsed["_raw_response"] = raw_response
                return parsed
                
            # If all fails, return empty dict (will trigger fallback)
            print(f"SLM failed to produce JSON. Raw: {raw_response[:200]}...")
            return {
                "mode": "unknown",
                "error": "No JSON found",
                "_raw_response": raw_response
            }
            
        except Exception as e:
            print(f"Ollama call failed: {e}")
            return {"mode": "unknown", "error": str(e)}


def _sanitize_enhanced_text(text: str) -> str:
    """Strip code fences/backticks while preserving line structure for readability."""
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
    coord_found = re.search(r"\([-+]?\d+\s*,\s*[-+]?\d+\)", rp)
    likely_mode = "vector_addition" if vector_like and coord_found else "generic_explainer"
    
    if likely_mode == "vector_addition":
        return (
            "TARGET_MODE: vector_addition\n"
            f"HIGH_LEVEL_INSTRUCTIONS: Visualize vector addition for {rp}.\n"
            "KEY_VALUES: vectors=[(1,2), (2,1)]; show_resultant=True\n"
            "REQUIREMENTS: standard layout."
        )
    else:
        return (
            "TARGET_MODE: generic_explainer\n"
            f"HIGH_LEVEL_INSTRUCTIONS: Explain {rp}.\n"
            "KEY_VALUES: sections=[Overview, Details]\n"
            "REQUIREMENTS: standard layout."
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
            # Resolve model name
            configured = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            legacy_map = {
                "gemini-1.5-flash": "gemini-flash-latest",
                "gemini-1.5-pro": "gemini-pro-latest",
            }
            resolved_model = legacy_map.get(configured, configured)
            system_instructions = (
                "You are the ENHANCER and MODE CLASSIFIER for a Math & Educational Animation Generator.\n\n"
                "The pipeline is:\n"
                "- USER PROMPT → YOU (Gemini) → ENHANCED SPEC\n"
                "- ENHANCED SPEC → SMALL MODEL (SLM) → JSON → Manim Animation\n\n"
                "Your job:\n"
                "1) Understand the user prompt.\n"
                "2) Decide which animation MODE fits best.\n"
                "3) Produce a clean, machine-readable description for the SLM.\n\n"
                "Supported modes (choose EXACTLY ONE):\n\n"
                "1. function_plot\n"
                "   - Plot f(x) on a 2D graph.\n"
                "   - Keywords: plot, graph, function, curve, sin(x), cos(x), polynomial, exponential, trig.\n\n"
                "2. parametric_plot\n"
                "   - Plot x(t), y(t) curves.\n"
                "   - Keywords: parametric, x(t), y(t), circle with cos(t), sin(t), spiral, Lissajous.\n\n"
                "3. vector_addition\n"
                "   - 2D vectors with arrows, tip-to-tail, resultant.\n"
                "   - Keywords: vector, vectors, arrow, resultant, add vectors.\n\n"
                "4. bubble_sort_visualization\n"
                "   - Show bars being sorted using bubble sort.\n"
        "   - Keywords: bubble sort, sort array, adjacent swaps.\n\n"
                "5. geometry_construction\n"
                "   - Points, lines, triangles, circles, bisectors, medians, perpendiculars.\n"
                "   - Keywords: triangle ABC, perpendicular bisector, angle bisector, construct, circle.\n\n"
                "6. matrix_visualization\n"
                "   - DEPRECATED: Use manim_code_gen instead for detailed matrix operations.\n"
                "   - For matrix multiplication, use manim_code_gen to show step-by-step dot products.\n\n"
                "7. number_line_interval\n"
                "   - Show inequalities on a number line (open/closed circles + shaded region).\n"
                "   - Keywords: inequality, number line, x > a, a < x <= b, interval.\n\n"
                "8. text_step_derivation\n"
                "   - Step-by-step derivation/proof with text lines.\n"
                "   - Keywords: derive, proof, step by step, show steps, formula derivation.\n\n"
                "9. generic_explainer\n"
                "   - Slide-style explanation (title + sections + bullet points).\n"
                "   - Use ONLY when the prompt is mainly conceptual/non-math or doesn’t fit any math modes.\n"
                "   - Examples: photosynthesis, history of WW2, OSI model, dark matter, etc.\n\n"
                "10. derivative_visualization\n"
                "   - Visualize slope, tangent line, and derivative curve.\n"
                "   - Keywords: derivative, tangent, slope, rate of change, f'(x).\n\n"
                "11. integral_area_visualization\n"
                "   - Visualize area under curve, Riemann sums, definite integral.\n"
                "   - Keywords: integral, area under curve, Riemann sum, accumulate.\n\n"
                "12. limit_visualization\n"
                "   - Visualize function approach to a point.\n"
                "   - Keywords: limit, approaches, tends to, x -> a.\n\n"
                "13. manim_code_gen\n"
                "   - Use for ANY animation that doesn't fit the other 12 modes (physics, complex motion, custom).\n"
                "   - Keywords: rolling ball, pendulum, projectile, brownian motion, custom.\n\n"
                "RULES:\n\n"
                "- ALWAYS choose a specific math mode if the prompt clearly asks for:\n"
                "  - a graph, vector, inequality, matrix, sort algorithm, geometry, derivation, derivative, integral, or limit.\n"
                "- Use manim_code_gen if the request is specific but doesn't fit existing modes (e.g. physics simulations).\n"
                "- Only choose generic_explainer when the topic is general theory or not easily visualised as math graphics.\n\n"
                "YOUR OUTPUT FORMAT (no extra text, no markdown):\n\n"
                "TARGET_MODE: <one of the 9 modes above, lowercase>\n\n"
                "HIGH_LEVEL_INSTRUCTIONS: <1–3 sentences describing what the animation should show>\n\n"
                "KEY_VALUES: <semi-structured key=value; key=value; ... with the critical data>\n\n"
                "REQUIREMENTS: <constraints like duration_seconds, colors, labels, etc.>\n"
            )
            few_shot = (
                "EXAMPLES:\n\n"
                "User: \"plot sin(x) from -2π to 2π and highlight peaks\"\n"
                "→\n"
                "TARGET_MODE: function_plot\n"
                "HIGH_LEVEL_INSTRUCTIONS: Plot sin(x) from -2π to 2π and highlight local maxima and minima.\n"
                "KEY_VALUES: function_expression=sin(x); x_min=-6.28; x_max=6.28; y_min=-2; y_max=2\n"
                "REQUIREMENTS: duration_seconds=8; title=\"Sine Wave with Extrema\"\n\n"
                "User: \"visualize −5 < x ≤ 7 on a number line\"\n"
                "→\n"
                "TARGET_MODE: number_line_interval\n"
                "HIGH_LEVEL_INSTRUCTIONS: Show the inequality -5 < x ≤ 7 on a number line.\n"
                "KEY_VALUES: inequality_string=-5 < x <= 7; left_value=-5; right_value=7; include_left=false; include_right=true\n"
                "REQUIREMENTS: duration_seconds=8; label_endpoints=true; shade_interval=true\n\n"
                "User: \"solve matrix multiplication of [[1,2],[3,4]] and [[5,6],[7,8]]\"\n"
                "→\n"
                "TARGET_MODE: matrix_visualization\n"
                "HIGH_LEVEL_INSTRUCTIONS: Show matrices A and B and animate the computation of their product C.\n"
                "KEY_VALUES: matrix_A=[[1,2],[3,4]]; matrix_B=[[5,6],[7,8]]; operation=multiplication; result_matrix=[[19,22],[43,50]]\n"
                "REQUIREMENTS: duration_seconds=15; title=\"Matrix Multiplication (2x2)\"; highlight_rows=true; highlight_columns=true; show_intermediate_steps=true\n\n"
                "User: \"explain photosynthesis\"\n"
                "→\n"
                "TARGET_MODE: generic_explainer\n"
                "HIGH_LEVEL_INSTRUCTIONS: Explain the process of photosynthesis for a high-school student.\n"
                "KEY_VALUES: topic=photosynthesis\n"
                "REQUIREMENTS: 3 sections; 3 bullet_points per section; duration_seconds=10\n"
            )
            model = genai.GenerativeModel(resolved_model, system_instruction=system_instructions)
            prompt = (
                f"{few_shot}\n\n"
                "Task: Rewrite the following input into the required format.\n\n"
                f"Input: {original}\n\n"
                "Output:"
            )
            resp = model.generate_content(prompt, generation_config={"temperature": 0.1})
            enhanced = getattr(resp, "text", "") or ""
            enhanced = _sanitize_enhanced_text(enhanced)
            
            if not enhanced or "TARGET_MODE:" not in enhanced:
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
                "Classify and plan animation. Output format:\n"
                "TARGET_MODE: <mode>\n"
                "HIGH_LEVEL_INSTRUCTIONS: <summary>\n"
                "KEY_VALUES: <data>\n"
                "REQUIREMENTS: <notes>\n"
                "Modes: function_plot, vector_addition, bubble_sort_visualization, parametric_plot, geometry_construction, matrix_visualization, text_step_derivation, number_line_interval, generic_explainer."
            )
            user_msg = f"Input: {original}\nOutput:"
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.1,
            )
            enhanced = resp["choices"][0]["message"]["content"].strip()
            enhanced = _sanitize_enhanced_text(enhanced)
            if not enhanced or "TARGET_MODE:" not in enhanced:
                return _fallback_structured(original), "fallback"
            return enhanced, "openai"
        except Exception:
            return _fallback_structured(original), "fallback"

    # No LLM configured
    return _fallback_structured(original), "fallback"


def validate_slm_instructions(payload: Dict[str, Any]) -> InstructionResponse:
    mode = payload.get("mode")
    # Use registry to check validity if possible, but we need strict schema validation here.
    # The registry has Pydantic models, so we could use them!
    
    if mode not in MODE_REGISTRY and mode != "unsupported":
        return InstructionResponse(status="error", message="Invalid JSON from SLM: unknown mode")

    if mode == "unsupported":
        return InstructionResponse(status="error", message="Unsupported prompt", mode=None, instructions=None)

    # Use the registry's Pydantic model for validation
    try:
        config_cls = MODE_REGISTRY[mode]
        # Pydantic validation
        validated_config = config_cls(**payload)
        # Return as dict
        return InstructionResponse(status="ok", mode=mode, instructions=validated_config.dict())
    except Exception as exc:
        return InstructionResponse(status="error", message=f"Invalid types for {mode}: {exc}")


def _parse_generic_explainer_from_text(text: str, title_hint: str) -> Dict[str, Any]:
    """
    Manually parse the enhanced text to extract sections for generic_explainer.
    Used when SLM fails to output valid JSON for non-math topics.
    """
    sections = []
    current_section = None
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers (heuristic: "Section X:" or just lines ending in ":")
        if (line.lower().startswith("section") and ":" in line) or (line.endswith(":") and len(line) < 50):
            if current_section:
                sections.append(current_section)
            
            heading = line.split(":", 1)[-1].strip()
            if not heading: 
                heading = line.replace(":", "").strip()
                
            current_section = {"heading": heading, "bullet_points": []}
        
        # Detect bullet points
        elif line.startswith("- ") or line.startswith("• "):
            content = line[2:].strip()
            if current_section:
                current_section["bullet_points"].append(content)
            else:
                # If no section yet, create a default one
                current_section = {"heading": "Overview", "bullet_points": [content]}
    
    if current_section:
        sections.append(current_section)
        
    # Fallback if parsing failed completely
    if not sections:
        sections = [
            {"heading": "Overview", "bullet_points": ["Could not parse details.", "Please try a more specific prompt."]}
        ]
        
    return {
        "mode": "generic_explainer",
        "title": title_hint,
        "sections": sections
    }


# Define stable math modes that use structured schemas (not raw code generation)
# Note: matrix_visualization removed to allow detailed step-by-step animations
MATH_STRUCTURED_MODES = {
    "function_plot",
    "number_line_interval",
    "bubble_sort_visualization",
    "derivative_visualization",
    "integral_area_visualization",
    "limit_visualization",
    "parametric_plot",
    "vector_addition",
    "scatter_points",
    "pythagoras_theorem",
    "geometry_construction",
}


async def call_slm_for_structured_mode(enhanced: str, target_mode: str) -> dict:
    """
    Call SLM to generate structured JSON for a specific math mode.
    Forbids manim_code_gen and enforces the target mode's schema.
    """
    # Add mode-specific examples to guide the SLM
    mode_examples = {
        "matrix_visualization": """
Example for matrix_visualization:
{
  "mode": "matrix_visualization",
  "title": "4x4 Matrix Multiplication",
  "matrixA": [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]],
  "matrixB": [[16,15,14,13], [12,11,10,9], [8,7,6,5], [4,3,2,1]],
  "animate_dot_products": true,
  "highlight_active_row_column": true,
  "duration_seconds": 22,
  "labels": {"matrixA": "A", "matrixB": "B", "matrixC": "C = A × B"}
}
""",
        "function_plot": """
Example for function_plot:
{
  "mode": "function_plot",
  "title": "Plot of x^2 - 5x + 1",
  "function_expression": "x**2 - 5*x + 1",
  "x_min": -2,
  "x_max": 7,
  "y_min": -6,
  "y_max": 15,
  "show_axes": true,
  "duration_seconds": 12
}
"""
    }
    
    example = mode_examples.get(target_mode, "")
    
    prompt = (
        f"{enhanced}\n\n"
        f"SYSTEM INSTRUCTION: You MUST output JSON with mode='{target_mode}' following its schema.\n"
        f"DO NOT use 'manim_code_gen'. Use the structured format for {target_mode}.\n"
        f"{example}\n"
        f"Output valid JSON only."
    )
    
    return await call_slm_with_ollama(prompt, target_mode=target_mode)


async def call_slm_for_manim_code_gen(enhanced: str, user_prompt: str) -> dict:
    """
    Call SLM to generate raw Manim Python code using the safe subset.
    """
    instr_match = re.search(r"HIGH_LEVEL_INSTRUCTIONS:\s*(.*)", enhanced)
    instruction = instr_match.group(1) if instr_match else user_prompt
    
    slm_prompt = (
        "You are a Python expert specializing in Manim animations.\n"
        "### RULES ###\n"
        "1. Supported imports ONLY: 'from manim import *', 'import math', 'import numpy as np'.\n"
        "2. Scene structure MUST be: 'class GeneratedScene(Scene): def construct(self):'.\n"
        "3. Use standard mobjects: Dot, Circle, Square, Line, VGroup, NumberPlane, Text.\n"
        "4. Use standard animations: Write, FadeIn, FadeOut, Rotate, MoveAlongPath, Transform.\n"
        "5. FORBIDDEN: 'MathTex', 'Tex', 'RateFunc', 'self.renderer.time', custom classes.\n"
        "6. NO LaTeX allowed. Use 'Text' for all formulas (e.g. Text('F = ma')).\n"
        "7. For continuous motion, use ValueTracker and updaters.\n"
        "8. For concepts: Use Text to explain, and simple shapes to illustrate.\n"
        "9. For PLOTTING: Use 'Axes' (not NumberPlane). Use 'axes.plot(lambda x: ...)'.\n"
        "10. FORBIDDEN: 'plot_point'. Use 'Dot(axes.coords_to_point(x, y))' instead.\n"
        "11. COORDINATES MUST BE 3D: [x, y, 0]. NEVER use 2D tuples like (x, y) for Arrow/Line.\n"
        "12. When using Axes: ALWAYS use 'axes.coords_to_point(x, y)' for positions.\n"
        "13. FORBIDDEN: 'ArcBetweenPoints'. Use 'Line' or 'Arrow' instead.\n"
        "14. UPDATERS: If using 'dt', signature must be 'def update(mob, dt):'. If not using dt, 'def update(mob):'.\n"
        "15. FORBIDDEN: 'get_axis_labels'. Manually create Text labels and place them next to axes.\n"
        "16. FORBIDDEN: 'include_numbers=True' in axis_config. ALWAYS use 'include_numbers=False' or omit it.\n"
        "17. MATRIX: Create with VGroup. Example: row=VGroup(*[Text(str(v), font_size=24) for v in [1,2,3]]); row.arrange(RIGHT, buff=0.3)\n"
        "18. MATRIX POSITION: matrix_A.shift(LEFT*4); matrix_B.next_to(matrix_A, RIGHT, buff=2)\n"
        "19. MATRIX ACCESS: matrix_A[row][col] to access cells. Use .set_color(YELLOW) to highlight.\n"
        "### TASK ###\n"
        f"Write a complete Manim script to: {instruction}\n"
        "Output ONLY the Python code, no explanations.\n"
    )
        
    print(f"DEBUG: Sending manim_code_gen prompt to Qwen 7B")
    return await call_slm_with_ollama(slm_prompt, target_mode="manim_code_gen")


@app.post("/generate/instructions", response_model=InstructionResponse)
async def generate_instructions(req: InstructionRequest) -> InstructionResponse:
    enhanced = ""
    enhanced_source = "fallback"
    
    try:
        # 1) Enhance the prompt using cloud LLM (structured format)
        enhanced, enhanced_source = await enhance_prompt_with_llm(req.prompt)
        print(f"DEBUG: Enhanced text:\n{enhanced}")
        
        # HARDCODED: Matrix multiplication shortcut
        if "matrix" in req.prompt.lower() and "multipl" in req.prompt.lower():
            print("DEBUG: HARDCODED matrix multiplication detected!")
            enhanced_text = (
                "TARGET_MODE: matrix_mult_detailed\n"
                "HIGH_LEVEL_INSTRUCTIONS: Create a detailed step-by-step animation of 4x4 matrix multiplication. "
                "Show matrices A and B side-by-side, highlight active row and column during each dot product calculation, "
                "display individual multiplications (e.g., 1×16=16, 2×12=24), show running sums, and fill result cells one by one.\n"
                "KEY_VALUES: matrix_size=4x4; visualization=step_by_step; show_dot_products=true; highlight_active=true\n"
                "REQUIREMENTS: duration_seconds=30; position_matrices_horizontally=true; show_computation_details=true"
            )
            return InstructionResponse(
                status="ok",
                mode="matrix_mult_detailed",
                instructions={
                    "mode": "matrix_mult_detailed", 
                    "title": "4x4 Matrix Multiplication - Step by Step",
                    "duration_seconds": 30.0,
                    "matrix_A": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                    "matrix_B": [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]],
                    "visualization_features": {
                        "show_matrices_side_by_side": True,
                        "highlight_active_row": True,
                        "highlight_active_column": True,
                        "show_individual_multiplications": True,
                        "show_running_sum": True,
                        "animate_result_filling": True,
                        "computation_details": "Display each multiplication (e.g., 1×16=16, 2×12=24) and sum"
                    },
                    "animation_style": {
                        "row_highlight_color": "YELLOW",
                        "column_highlight_color": "GREEN",
                        "result_color": "RED",
                        "computation_font_size": 18,
                        "matrix_font_size": 24
                    },
                    "computed_cells": "First 2x2 cells (4 cells total) for demonstration"
                },
                message="Using hardcoded matrix_mult_detailed mode for optimal visualization",
                enhanced_prompt=enhanced_text,
                enhanced_source="matrix_mult_detailed_mode"
            )

        
        # Extract TARGET_MODE from enhanced text
        target_mode_match = re.search(r"TARGET_MODE:\s*(\w+)", enhanced)
        target_mode = target_mode_match.group(1) if target_mode_match else "manim_code_gen"
        print(f"DEBUG: Extracted TARGET_MODE: '{target_mode}'")
        
        # BRANCH: Structured Math Modes vs General Manim Code Gen
        if target_mode in MATH_STRUCTURED_MODES:
            print(f"DEBUG: Using STRUCTURED MODE path for: {target_mode}")
            slm_json = await call_slm_for_structured_mode(enhanced, target_mode)
        else:
            print(f"DEBUG: Using MANIM_CODE_GEN path for: {target_mode}")
            slm_json = await call_slm_for_manim_code_gen(enhanced, req.prompt)

        print(f"DEBUG: SLM JSON keys: {slm_json.keys()}")
        
        # 3) Validate Mode Consistency
        slm_mode = slm_json.get("mode")
        
        # If SLM chose generic_explainer but TARGET_MODE was specific (and supported), retry once.
        if slm_mode == "generic_explainer" and target_mode != "generic_explainer" and target_mode in MODE_REGISTRY:
             print(f"Warning: SLM returned generic_explainer but TARGET_MODE was {target_mode}. Retrying with forced constraint.")
             
             # Append a strict instruction to the enhanced prompt
             forced_prompt = enhanced + f"\n\nSYSTEM ALERT: You previously ignored the TARGET_MODE. You MUST generate JSON for mode: '{target_mode}'. DO NOT generate generic_explainer."
             
             # Retry call
             slm_json = await call_slm_with_ollama(forced_prompt)
             slm_mode = slm_json.get("mode")
             
             # If still failing, we enforce TARGET_MODE by synthesizing a config
             if slm_mode != target_mode:
                 print(f"Error: SLM failed to produce {target_mode} even after retry. Got {slm_mode}. Synthesizing config.")
                 # Synthesize config from KEY_VALUES
                 slm_json = _synthesize_config_from_key_values(target_mode, enhanced, req.prompt)

        # Validate schema
        validated = validate_slm_instructions(slm_json)
        if validated.status == "ok":
            validated.enhanced_prompt = enhanced
            validated.enhanced_source = enhanced_source
            return validated
        
        # If validation failed, check if we can synthesize a valid config for a math mode
        if target_mode in MODE_REGISTRY and target_mode not in ("generic_explainer", "manim_code_gen"):
             print(f"Validation failed for {target_mode}. Attempting synthesis.")
             synthesized_json = _synthesize_config_from_key_values(target_mode, enhanced, req.prompt)
             validated = validate_slm_instructions(synthesized_json)
             if validated.status == "ok":
                 validated.enhanced_prompt = enhanced
                 validated.enhanced_source = enhanced_source
                 return validated

        raise ValueError(f"Validation failed: {validated.message}")

    except Exception as exc:
        print(f"SLM/Validation failed ({exc}). Returning Rolling Ball fallback.")
        code = (
            "from manim import *\n\n"
            "class GeneratedScene(Scene):\n"
            "    def construct(self):\n"
            "        # Fallback Animation: Rolling Ball\n"
            "        ground = Line(LEFT * 5, RIGHT * 5)\n"
            "        ball = Circle(radius=1.0).shift(LEFT * 4 + UP * 1.0)\n"
            "        ball.set_fill(BLUE, opacity=0.8)\n"
            "        ball.set_stroke(WHITE, width=2)\n"
            "        path = Line(ball.get_center(), ball.get_center() + RIGHT * 8)\n"
            "        path.set_stroke(YELLOW, width=2)\n"
            "        rim_dot = Dot(ball.point_at_angle(PI / 2), color=RED)\n"
            "        rolling_group = VGroup(ball, rim_dot)\n"
            "        self.play(Create(ground))\n"
            "        self.play(Create(ball), FadeIn(rim_dot))\n"
            "        self.play(Create(path))\n"
            "        self.play(\n"
            "            rolling_group.animate.shift(RIGHT * 8).rotate(-8, about_point=ball.get_center()),\n"
            "            run_time=6,\n"
            "            rate_func=linear\n"
            "        )\n"
            "        self.wait(1)\n"
        )
        return InstructionResponse(
            status="ok",
            mode="manim_code_gen",
            instructions={
                "mode": "manim_code_gen",
                "code": code,
                "title": "Rolling Ball (Fallback)",
            },
            enhanced_prompt=enhanced if 'enhanced' in locals() else "",
            enhanced_source="fallback",
        )




def _synthesize_config_from_key_values(mode: str, enhanced_text: str, user_prompt: str) -> Dict[str, Any]:
    """
    Synthesize a minimal valid config for a specific mode by parsing KEY_VALUES from enhanced text.
    This is a fail-safe when SLM refuses to generate the correct mode.
    """
    # Extract KEY_VALUES block
    kv_match = re.search(r"KEY_VALUES:\s*(.*)", enhanced_text)
    kv_str = kv_match.group(1) if kv_match else ""
    
    # Simple parser for key=value; key=value
    kv_pairs = {}
    if kv_str:
        # Split by semicolon
        parts = kv_str.split(';')
        for part in parts:
            if '=' in part:
                k, v = part.split('=', 1)
                kv_pairs[k.strip()] = v.strip()
    
    # Extract title
    # Try multiple patterns
    title = None
    # Pattern 1: title="Something" in REQUIREMENTS or KEY_VALUES
    t_match = re.search(r'title=["\']([^"\']+)["\']', enhanced_text, re.IGNORECASE)
    if t_match:
        title = t_match.group(1)
    
    # Pattern 2: Title: "Something" (Generic explainer style)
    if not title:
        t_match = re.search(r'Title:\s*["\']([^"\']+)["\']', enhanced_text, re.IGNORECASE)
        if t_match:
            title = t_match.group(1)
            
    # Fallback to user prompt (cleaned)
    if not title:
        # Take first 50 chars of prompt, clean up
        clean_prompt = re.sub(r'[^\w\s-]', '', user_prompt).strip()
        title = clean_prompt[:50] if clean_prompt else "Animation"

    # Build config based on mode
    # Build config based on mode
    if mode == "number_line_interval":
        # Defaults
        left_val = None
        right_val = None
        try:
            if "left_value" in kv_pairs:
                left_val = float(kv_pairs["left_value"])
            if "right_value" in kv_pairs:
                right_val = float(kv_pairs["right_value"])
            
            # Fallback to old keys if new ones missing
            if left_val is None and "start_value" in kv_pairs:
                left_val = float(kv_pairs["start_value"])
            if right_val is None and "end_value" in kv_pairs:
                right_val = float(kv_pairs["end_value"])
        except: pass
        
        return {
            "mode": "number_line_interval",
            "interval_type": kv_pairs.get("interval_type", "<"),
            "left_value": left_val,
            "right_value": right_val,
            "include_left": kv_pairs.get("include_left", "false").lower() == "true",
            "include_right": kv_pairs.get("include_right", "true").lower() == "true",
            "title": title,
            "duration_seconds": 8.0
        }
    
    elif mode == "function_plot":
        expr = kv_pairs.get("function_expression", "x")
        return {
            "mode": "function_plot",
            "function_expression": expr,
            "x_min": float(kv_pairs.get("x_min", -5)),
            "x_max": float(kv_pairs.get("x_max", 5)),
            "y_min": -5.0,
            "y_max": 5.0,
            "duration_seconds": 8.0,
            "title": title
        }

    elif mode == "matrix_visualization":
        import ast
        try:
            matrix_a_str = kv_pairs.get("matrix", kv_pairs.get("matrix_a", "[[1,0],[0,1]]"))
            matrix_a = ast.literal_eval(matrix_a_str)
            
            matrix_b = None
            if "matrix_b" in kv_pairs:
                matrix_b = ast.literal_eval(kv_pairs["matrix_b"])
            
            operation = kv_pairs.get("operation", None)
            
            return {
                "mode": "matrix_visualization",
                "matrix": matrix_a,
                "matrix_b": matrix_b,
                "operation": operation,
                "duration_seconds": 12.0,
                "title": title
            }
        except Exception as e:
            print(f"Matrix synthesis failed: {e}")
            # Fallback to identity matrix if parsing fails
            return {
                "mode": "matrix_visualization",
                "matrix": [[1, 0], [0, 1]],
                "title": "Matrix Visualization (Fallback)"
            }

    elif mode == "derivative_visualization":
        return {
            "mode": "derivative_visualization",
            "function_expression": kv_pairs.get("function_expression", "x**2"),
            "x_min": float(kv_pairs.get("x_min", -5)),
            "x_max": float(kv_pairs.get("x_max", 5)),
            "x0": float(kv_pairs.get("x0", 0)),
            "show_derivative_curve": kv_pairs.get("show_derivative_curve", "true").lower() == "true",
            "duration_seconds": 8.0,
            "title": title
        }
    elif mode == "integral_area_visualization":
        return {
            "mode": "integral_area_visualization",
            "function_expression": kv_pairs.get("function_expression", "x**2"),
            "a": float(kv_pairs.get("a", 0)),
            "b": float(kv_pairs.get("b", 1)),
            "num_rectangles": int(kv_pairs.get("num_rectangles", 20)),
            "method": kv_pairs.get("method", "midpoint"),
            "show_exact_area_label": kv_pairs.get("show_exact_area_label", "true").lower() == "true",
            "duration_seconds": 10.0,
            "title": title
        }
    elif mode == "limit_visualization":
        return {
            "mode": "limit_visualization",
            "function_expression": kv_pairs.get("function_expression", "x**2"),
            "x_min": float(kv_pairs.get("x_min", -5)),
            "x_max": float(kv_pairs.get("x_max", 5)),
            "x0": float(kv_pairs.get("x0", 0)),
            "show_left_right": kv_pairs.get("show_left_right", "true").lower() == "true",
            "duration_seconds": 8.0,
            "title": title
        }

    # Add other modes as needed...
    
    # If we can't synthesize, return generic_explainer structure but labeled as the target mode 
    # (which will likely fail validation if schema is strict, so we fallback to generic)
    return _parse_generic_explainer_from_text(enhanced_text, title)


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
