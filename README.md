# NTA Question Extractor & LaTeX Digitizer

A deterministic pipeline that extracts **NTA JEE/NEET exam questions** from scanned images, converts math to **LaTeX**, and cuts out **circuit diagrams, ray diagrams, and chemistry reaction schemes** with pixel-perfect precision.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **LaTeX OCR** | Extracts question text + all 4 options as clean LaTeX |
| ✂️ **Cut-to-cut cropping** | Pixel-exact diagram extraction — no surrounding text |
| 🎨 **Custom styling** | Navy blue / transparent / dark background modes |
| 🔌 **Dual backend** | Cloud (Gemini API, free tier) **or** Local (Ollama on GPU) |
| 🔁 **Resume support** | Batch processing with checkpoint — survives quota limits |
| ⚡ **Zero GPU required** | Cloud mode works on any laptop using the free Gemini API |

---

## 📁 Project Structure

```
questions/
├── test_latex_extractor.py      # Single-image CLI tool (cloud API)
├── batch_latex_extractor.py     # Batch processing (cloud API, 2994 questions)
├── extract_latex_local.py       # Local GPU mode via Ollama
├── run_extractor.py             # NTA dataset downloader
├── preview_server.py            # Local web viewer server
├── question_viewer.html         # Web UI for browsing questions
├── nta_extractor/               # NTA scraping modules
├── all_nta_questions.json       # Full dataset (2,999 questions)
└── downloads/
    ├── question_images/         # Downloaded exam images (~3000 JPG/PNG)
    └── diagrams/                # Extracted cut-to-cut diagrams
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/rajatkolhapure/nta-latex-extractor.git
cd nta-latex-extractor
pip install opencv-python numpy requests pillow
```

### 2. Choose a Backend

| Backend | Best For | Cost | Setup |
|---|---|---|---|
| ☁️ **Gemini API (Cloud)** | Quick start, any laptop | Free (20 req/day) | Just an API key |
| 🖥️ **Ollama (Local GPU)** | Unlimited, private | Free forever | RTX 4070+ recommended |

---

## ☁️ Option A — Cloud API (Gemini, Free Tier)

Get a free API key at **[aistudio.google.com](https://aistudio.google.com)** (no credit card needed).

### Test a single image

```bash
python test_latex_extractor.py \
  --image "downloads/question_images/14_20191003150620.JPG" \
  --gemini-key "YOUR_API_KEY_HERE" \
  --bg-mode white \
  --stroke-color "#1e40af"
```

### Run batch on all questions

```bash
python batch_latex_extractor.py \
  --gemini-key "YOUR_API_KEY_HERE" \
  --bg-mode white \
  --stroke-color "#1e40af"
```

> **Note:** Free tier allows ~20 requests/day on `gemini-2.5-flash`. The script automatically falls back to `gemini-2.5-flash-lite` (higher quota) and resumes from where it left off each day.

---

## 🖥️ Option B — Local GPU (Ollama + Qwen2.5-VL)

Run **completely offline** on your RTX 4070 laptop GPU. No API key, no rate limits, no data leaves your machine.

### Recommended Model

| Model | VRAM | Speed | Quality |
|---|---|---|---|
| **`qwen2.5-vl:7b`** ✅ Recommended | ~6 GB | ~3–5 sec/img | Excellent math OCR |
| `qwen2.5-vl:3b` | ~3 GB | ~1–2 sec/img | Good |
| `llava:13b` | ~8 GB | ~6–8 sec/img | Good |

> A **4070 laptop GPU (8 GB VRAM)** runs `qwen2.5-vl:7b` comfortably with room to spare.

---

### Step 1 — Install Ollama

Download and install from **[ollama.com](https://ollama.com/download)**:

- **Windows**: Download the `.exe` installer and run it
- **Linux/Mac**: `curl -fsSL https://ollama.com/install.sh | sh`

Verify:
```bash
ollama --version
```

---

### Step 2 — Download the Vision Model

```bash
ollama pull qwen2.5-vl:7b
```

This downloads ~4.7 GB. Progress is shown automatically. The model is saved to `~/.ollama/models/` and persists across reboots.

Verify the model is ready:
```bash
ollama list
# Should show: qwen2.5-vl:7b
```

---

### Step 3 — Start the Ollama Server

```bash
ollama serve
```

> Ollama runs on `http://localhost:11434`. On Windows, the Ollama tray app starts it automatically after installation. Verify it's running:

```bash
curl http://localhost:11434
# Should print: Ollama is running
```

---

### Step 4 — Run Local Extraction

**Test on 5 questions first:**
```bash
python extract_latex_local.py --limit 5
```

**Run on the full dataset:**
```bash
python extract_latex_local.py
```

Output is saved to `all_nta_questions_latex.json` after every question — safe to interrupt and resume anytime.

---

### GPU Monitoring (Optional)

Watch GPU utilisation while running:
```bash
nvidia-smi -l 1
```

Expected: ~6 GB VRAM usage, ~80–100% GPU load during inference.

---

## 🎨 Diagram Output Styles

Control diagram appearance with `--bg-mode` and `--stroke-color`:

| Mode | Command | Result |
|---|---|---|
| White background (default) | `--bg-mode white --stroke-color "#1e40af"` | Navy ink on white |
| Transparent PNG | `--bg-mode transparent --stroke-color "#1e40af"` | Navy ink, transparent bg |
| Dark mode | `--bg-mode dark` | White ink on slate bg |
| Custom color | `--stroke-color "#dc2626"` | Red ink |

---

## 🧠 How Diagram Detection Works

The CV pipeline is **fully deterministic** — no AI needed for cropping:

1. **Semantic filter** — checks question text for visual keywords (`figure`, `circuit`, `shown in`, `reaction`, `product`, etc.). Rejects pure-math questions before any image processing.
2. **Row projection** — finds contiguous ink blocks by summing dark pixels per row.
3. **Text paragraph rejection** — blocks with >85% horizontal coverage AND low row-ink variance are rejected as typeset text (not diagrams).
4. **Satellite label expansion** — merges adjacent narrow blocks (apex labels like `P`, dimension arrows like `←d/2→`) into the diagram crop.
5. **Cut-to-cut crop** — final bounding box is pixel-tight on the ink strokes.
6. **Color transform** — applies custom stroke color over a clean background.

---

## 📊 Dataset

- **2,999 questions** from NTA JEE Main papers (2019–2022)
- **Subjects**: Physics, Chemistry, Mathematics
- **~2,994 images** downloaded successfully
- Source: [nta.ac.in](https://nta.ac.in)

To re-download the dataset:
```bash
python run_extractor.py
```

---

## 📋 Output Schema

Each processed question in `all_extracted_latex_questions.json`:

```json
{
  "paperId": "133",
  "paperTitle": "PAPER 1 09-09-2020 MORNING",
  "subject": "Physics",
  "questionNumber": 14,
  "latexQuestion": "A capacitor of capacitance $C_0$ is charged to a potential $V_0$...",
  "latexOptions": [
    {"key": "1", "latex": "$\\frac{3}{2} C_0 V_0^2$"},
    {"key": "2", "latex": "$\\frac{1}{4} C_0 V_0^2$"},
    {"key": "3", "latex": "$\\frac{3}{4} C_0 V_0^2$"},
    {"key": "4", "latex": "$\\frac{1}{2} C_0 V_0^2$"}
  ],
  "hasDiagram": true,
  "diagramBBox": [15, 149, 232, 319],
  "diagramLocalPath": "./downloads/diagrams/cuttocut_14_20191003150620.png"
}
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Gemini API 429` | Daily quota hit — script auto-retries next day. Or switch to local mode. |
| `Ollama not running` | Run `ollama serve` in a terminal, or restart the Ollama Windows tray app. |
| `CUDA out of memory` | Try `qwen2.5-vl:3b` instead — uses only ~3 GB VRAM. |
| `Image not found` | Run `python run_extractor.py` to re-download missing images. |
| `JSON parse error` | The `fix_latex_json` parser handles this automatically — report the image path if it still fails. |

---

## 📄 License

MIT © Rajat Kolhapure
