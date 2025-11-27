from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
import math
import numpy as np
import re
from pydantic import BaseModel, Field, validator


# Limited, safe namespace for evaluating math expressions
_ALLOWED_NAMES: Dict[str, Any] = {
    # constants (NumPy for array-friendly ops)
    "pi": np.pi,
    "e": np.e,
    # basic functions (NumPy ufuncs to support scalars and arrays)
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,  # natural log
    "sqrt": np.sqrt,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "pow": np.power,
}


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug from a title."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "animation"


def build_function_callable(expression: str) -> Callable[[float], float]:
    """Return a safe callable f(x) for the given math expression string.

    Security considerations:
    - No builtins, only a small, explicit allowlist in _ALLOWED_NAMES.
    - Only variable allowed is `x`.
    - Convenience: replace caret '^' with Python power '**'.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("function_expression must be a non-empty string")

    expr = expression.replace("^", "**").strip()

    # Precompile to validate early
    code = compile(expr, filename="<function_expression>", mode="eval")

    def f(x):  # accept float or numpy array
        return eval(code, {"__builtins__": {}}, {**_ALLOWED_NAMES, "x": x})

    # Quick sanity check at a couple points to surface obvious errors early
    try:
        test0 = f(0.0)
        test1 = f(1.0)
        # force evaluation to scalar for sanity check only
        float(np.asarray(test0).reshape(-1)[0])
        float(np.asarray(test1).reshape(-1)[0])
    except Exception as exc:
        raise ValueError(f"Invalid function expression '{expression}': {exc}") from exc

    return f


@dataclass
class FunctionPlotConfig:
    """Typed config for the 'function_plot' mode.

    This is intentionally minimal for STEP 1, but structured to scale later.
    """

    mode: str
    function_expression: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    duration_seconds: float
    title: str

    # Derived field: built callable from the expression
    function_callable: Callable[[float], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode != "function_plot":
            raise ValueError("Unsupported mode: expected 'function_plot'")
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be < x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be < y_max")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        self.function_callable = build_function_callable(self.function_expression)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionPlotConfig":
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
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required keys in config: {', '.join(missing)}")

        return cls(
            mode=str(data["mode"]),
            function_expression=str(data["function_expression"]),
            x_min=float(data["x_min"]),
            x_max=float(data["x_max"]),
            y_min=float(data["y_min"]),
            y_max=float(data["y_max"]),
            duration_seconds=float(data["duration_seconds"]),
            title=str(data["title"]),
        )


# STEP 3: Vector addition mode config models (Pydantic)

class Vector2D(BaseModel):
    label: str = Field(..., description="Vector label, e.g., 'v1'")
    x: float = Field(..., description="X component")
    y: float = Field(..., description="Y component")


class VectorAdditionConfig(BaseModel):
    mode: str = Field("vector_addition", description="Must be 'vector_addition'")
    vectors: List[Vector2D] = Field(..., min_items=1, description="List of 2D vectors")
    show_resultant: bool = Field(True, description="Whether to draw resultant vector")
    show_tip_to_tail: bool = Field(True, description="Whether to animate tip-to-tail placement")
    title: Optional[str] = Field("Vector Addition", description="Title shown atop the scene")

    @validator("mode")
    def check_mode(cls, v: str) -> str:
        if v != "vector_addition":
            raise ValueError("Unsupported mode: expected 'vector_addition'")
        return v
