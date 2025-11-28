# Math Animation Generator (Manim + FastAPI)

An intelligent educational animation generator that creates step-by-step mathematical visualizations using Manim. Powered by AI (Gemini + Qwen) for prompt enhancement and code generation.

## 🎯 Features

- **15+ Animation Modes**: Function plots, matrix operations, vector addition, calculus visualizations, sorting algorithms, and more
- **AI-Powered Generation**: Uses Gemini for prompt enhancement and Qwen for Manim code generation
- **Hardcoded Matrix Multiplication**: Instant 4x4 matrix multiplication with step-by-step dot product visualization
- **Safe Code Generation**: 19 safety rules ensure generated Manim code always runs
- **Interactive Web UI**: React frontend with Manual and Prompt modes
- **Dual Pipeline**: Structured modes for math (stable) + manim_code_gen for concepts (flexible)

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Node.js 18+** (for frontend)
3. **Manim** (animation library)
4. **Ollama** (local LLM runtime)
5. **Gemini API Key** (for prompt enhancement)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/nipun172006/AnimationGenerator.git
cd AnimationGenerator
```

#### 2. Backend Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Manim
pip install manim

# Set up environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="gemini-2.0-flash-exp"
```

#### 3. Install Ollama and Models
```bash
# Install Ollama (macOS)
brew install ollama

# Pull required models
ollama pull gemma3:4b
ollama pull qwen2.5-coder:7b

# Start Ollama server
ollama serve
```

#### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

#### 5. Start Backend
```bash
# In project root
uvicorn app:app --reload
```

## 📖 Usage

### Web Interface

1. Open `http://localhost:5173` (frontend)
2. Choose **Prompt Mode** or **Manual Mode**

#### Prompt Mode (Recommended)
- Type: `"show matrix multiplication"`
- Click **Generate Instructions**
- Click **Render Animation**
- Video saved to `outputs/videos/`

#### Manual Mode
- Select mode (e.g., `function_plot`)
- Fill in parameters
- Click **Render**

### API Usage

#### Generate Instructions
```bash
curl -X POST http://localhost:8000/generate/instructions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "show matrix multiplication"}'
```

#### Render Animation
```bash
curl -X POST http://localhost:8000/render/any_mode \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "matrix_mult_detailed",
    "payload": {
      "mode": "matrix_mult_detailed",
      "title": "4x4 Matrix Multiplication"
    }
  }'
```

## 🎨 Supported Animation Modes

### Math Structured Modes (Stable)
- `function_plot` - Plot mathematical functions
- `parametric_plot` - Parametric curves (circles, spirals)
- `derivative_visualization` - Show tangent lines and slopes
- `integral_area_visualization` - Area under curves
- `limit_visualization` - Function limits
- `number_line_interval` - Inequalities on number line
- `vector_addition` - 2D vector operations
- `scatter_points` - Data point visualization
- `pythagoras_theorem` - Geometric proof
- `geometry_construction` - Compass and straightedge
- `bubble_sort_visualization` - Sorting algorithm
- `matrix_mult_detailed` - **4x4 matrix multiplication with step-by-step dot products**

### Flexible Modes
- `manim_code_gen` - AI-generated Manim code for any concept
- `generic_explainer` - Slide-style explanations

## 🔧 Configuration

### Environment Variables

Create `.env` file or export to shell:

```bash
# Required
export GEMINI_API_KEY="AIzaSy..."
export GEMINI_MODEL="gemini-2.0-flash-exp"

# Optional (defaults shown)
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="gemma3:4b"
export MANIM_MODEL="qwen2.5-coder:7b"
```

### Manim Settings

Edit `app.py` to change:
- Video quality: `config.quality = "low_quality"` (480p15) or `"high_quality"` (1080p60)
- Output directory: `outputs/videos/`

## 🧠 Architecture

### Pipeline Flow

```
User Prompt
    ↓
Gemini Enhancement (TARGET_MODE detection)
    ↓
Branch Decision
    ├─→ Math Structured Mode → Stable Renderer
    └─→ manim_code_gen → Qwen 7B → Sanitizers → Manim
```

### Key Components

1. **Prompt Enhancement** (`enhance_prompt_with_llm`)
   - Uses Gemini to classify prompt
   - Extracts TARGET_MODE, KEY_VALUES, REQUIREMENTS

2. **Mode Branching** (`generate_instructions`)
   - Math prompts → Structured modes
   - Concepts → `manim_code_gen`

3. **Code Generation** (`call_slm_for_manim_code_gen`)
   - 19 safety rules for Manim
   - Matrix positioning, LaTeX avoidance, 3D coordinates

4. **Sanitizers** (`manim_code_gen_mode.py`)
   - `sanitize_manim_code_for_latex` - Remove LaTeX
   - `sanitize_matrix_code` - Fix matrix errors
   - `sanitize_axes_numbers` - Prevent LaTeX crashes

## 🎯 Matrix Multiplication Example

**Prompt:** `"show matrix multiplication"`

**Generated Animation:**
- Two 4x4 matrices (A and B) displayed side-by-side
- Row in A highlighted in YELLOW
- Column in B highlighted in GREEN
- Individual multiplications shown: `1×16=16`, `2×12=24`, etc.
- Running sum displayed: `Sum = 80`
- Result cell filled in RED
- Computes first 2×2 cells (4 total) for demonstration

**Output:** `outputs/videos/matrix_mult_detailed.mp4`

## 📁 Project Structure

```
AnimationGenerator/
├── app.py                          # FastAPI backend
├── animation_config.py             # Pydantic schemas
├── animation_modes/
│   ├── __init__.py                # Mode registry
│   ├── base.py                    # Base config
│   ├── function_plot_mode.py
│   ├── matrix_mult_detailed_mode.py  # Hardcoded matrix mode
│   ├── manim_code_gen_mode.py     # AI code generation
│   └── ...                        # Other modes
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # React UI
│   │   └── main.tsx
│   └── package.json
├── outputs/
│   └── videos/                   # Generated animations
├── dataset.json                  # 65 training examples
└── README.md
```

## 🛡️ Safety Features

### Manim Code Generation Rules

1. **Imports**: Only `manim`, `math`, `numpy`
2. **No LaTeX**: Use `Text` instead of `MathTex`/`Tex`
3. **3D Coordinates**: `[x, y, 0]` not `(x, y)`
4. **Matrix Creation**: `VGroup` with proper arrangement
5. **Positioning**: `.shift()`, `.next_to()` for spacing
6. **Forbidden APIs**: `ArcBetweenPoints`, `plot_point`, `RateFunc`
7. **Updater Signatures**: Correct `dt` parameter
8. **Axis Configuration**: `include_numbers=False`

### Sanitizers

- **LaTeX Removal**: Replaces `MathTex` with `Text`
- **Matrix Fixes**: Corrects `.scale()` on strings
- **Arc Replacement**: `ArcBetweenPoints` → `Line`
- **Axis Labels**: Comments out `get_axis_labels`

## 🐛 Troubleshooting

### Ollama Not Running
```bash
ollama serve
```

### Manim Not Found
```bash
pip install manim
```

### Port 8000 Already in Use
```bash
lsof -i :8000 | grep Python | awk '{print $2}' | xargs kill
```

### Frontend Not Loading
```bash
cd frontend
npm install
npm run dev
```

## 📊 Performance

- **Prompt Enhancement**: ~2-3s (Gemini)
- **Code Generation**: ~5-10s (Qwen 7B)
- **Manim Rendering**: ~10-30s (depends on complexity)
- **Total**: ~20-45s per animation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Open Pull Request

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Manim** - Animation engine by 3Blue1Brown
- **Gemini** - Google's multimodal AI
- **Qwen 2.5 Coder** - Alibaba's code generation model
- **Ollama** - Local LLM runtime

## 📧 Contact

- **Author**: Nipun Thumu
- **GitHub**: [@nipun172006](https://github.com/nipun172006)
- **Repository**: [AnimationGenerator](https://github.com/nipun172006/AnimationGenerator)

---

**Made with ❤️ for education and mathematics visualization**
