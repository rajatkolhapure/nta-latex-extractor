# NTA Question Extractor & LaTeX Digitizer

An automated engineering pipeline that extracts NTA JEE, NEET, and MHT-CET exam questions from scanned images and PDF question papers. The pipeline digitizes mathematical expressions into LaTeX, transcribes observation tables into semantic HTML, cuts out visual diagrams (circuits, ray diagrams, chemical structures) with pixel-exact cropping, and exports standardized JSON datasets.

---

## Key Capabilities

- **Mathematical LaTeX Transcription**: Converts complex mathematical symbols, Greek notation, matrices, roots, fractions, and chemical reaction notation into LaTeX expressions.
- **Deterministic Diagram Extraction**: Combines semantic text triggers with computer vision ink-projection analysis to crop diagrams cleanly without surrounding text lines.
- **Sub-pixel Diagram Upscaling & Anti-aliasing**: Upscales diagram crops by 2.5x using Lanczos4 interpolation with slight Gaussian smoothing and threshold-based stroke isolation.
- **Dynamic Subject Routing**: Automatically routes extracted questions into subject-specific datasets (Physics, Chemistry, Mathematics).
- **Dual Processing Engines**: Supports cloud vision models (Gemini 3.5 Flash Lite with automated fallback chain) and offline GPU acceleration (Ollama with Qwen2.5-VL).
- **Stateful Resumption**: Automatically tracks processed records and skips previously completed questions across interruptions and rate limits.
- **Interactive Question Explorer**: Local browser-based UI with live KaTeX math rendering, observation table formatting, dark/light theme switching, and diagram inspection.

---

## Directory Structure

```
questions/
|-- cet/                         # Directory containing question paper PDFs
|-- data/                        # Datasets (raw, split subjects, and extracted JSONs)
|   |-- all_nta_questions.json
|   |-- all_nta_questions_sorted.json
|   |-- chemistry_questions.json
|   |-- extracted_chemistry.json
|   |-- extracted_mathematics.json
|   |-- extracted_physics.json
|   |-- mathematics_questions.json
|   |-- nta_papers.json
|   |-- physics_questions.json
|   |-- questions_by_subject.json
|   `-- test_q30.json
|-- downloads/                   # Downloaded assets and generated media
|   |-- diagrams/                # Cropped transparent/white stroke diagrams
|   |-- pdf_pages/               # Rendered high-DPI page images from PDFs
|   `-- question_images/         # Raw question images downloaded from NTA
|-- extractors/                  # Extraction engines
|   |-- batch_latex_extractor.py # Batch image-to-LaTeX extractor
|   |-- pdf_latex_extractor.py   # PDF question paper extractor
|   |-- extract_latex_local.py   # Offline GPU extractor via Ollama
|   `-- nta_extractor/           # NTA mock test portal scraper package
|       |-- classifier.py
|       |-- cli.py
|       |-- config.py
|       |-- downloader.py
|       |-- filter.py
|       |-- models.py
|       `-- engines/
|-- tests/                       # Evaluation and test harnesses
|   `-- test_latex_extractor.py  # Single-image test script
|-- tools/                       # Utility scripts
|   |-- run_extractor.py         # NTA scraper CLI runner
|   `-- sort_by_subject.py       # Subject-wise splitter and sorter
|-- viewer/                      # Browser inspection interface
|   |-- preview_server.py        # Local HTTP server with asset routing
|   `-- question_viewer.html     # KaTeX question viewer application
|-- requirements.txt             # Python dependency specifications
|-- README.md                    # Project documentation
`-- .gitignore                   # Git exclusion rules
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

```bash
git clone https://github.com/rajatkolhapure/nta-latex-extractor.git
cd nta-latex-extractor
pip install -r requirements.txt
```

---

## Usage Guide

### 1. PDF Question Paper Extraction

Extract questions and diagrams directly from question paper PDFs (e.g., MHT-CET, JEE papers in `cet/`):

```bash
python extractors/pdf_latex_extractor.py --pdf-dir cet --gemini-key "YOUR_GEMINI_API_KEY"
```

Options:
- `--pdf`: Path to a single PDF file (overrides `--pdf-dir`).
- `--pdf-dir`: Directory containing PDF files (default: `cet`).
- `--gemini-key`: Google Gemini API key.
- `--stroke-color`: Stroke color in hex format (default: `#ffffff`).
- `--bg-mode`: Background mode (`transparent`, `white`, `dark`).
- `--dpi`: Rendering resolution (default: `180`).
- `--start-page`: Starting page index (default: `1`).
- `--end-page`: Ending page index (`0` for all pages).

### 2. Batch Image Dataset Extraction

Process scraped question images from a JSON dataset:

```bash
python extractors/batch_latex_extractor.py \
  --gemini-key "YOUR_GEMINI_API_KEY" \
  --subject Physics \
  --input-json physics_questions.json \
  --stroke-color "#ffffff" \
  --bg-mode transparent
```

Options:
- `--gemini-key`: Google Gemini API key.
- `--subject`: Subject filter (`Physics`, `Chemistry`, `Mathematics`).
- `--input-json`: Input dataset file inside `data/` or relative path.
- `--output-json`: Target output file (automatically routed to `data/extracted_<subject>.json`).
- `--limit`: Maximum number of questions to process (`0` for all).
- `--stroke-color`: Output diagram ink color (default: `#ffffff`).
- `--bg-mode`: Output diagram background (`transparent`, `white`, `dark`).

### 3. Single Image Testing

Test math transcription and diagram cropping on an individual question image:

```bash
python tests/test_latex_extractor.py \
  --image downloads/question_images/14_20191003150620.JPG \
  --gemini-key "YOUR_GEMINI_API_KEY" \
  --bg-mode transparent \
  --stroke-color "#ffffff"
```

### 4. Offline Local GPU Extraction (Ollama)

Run extraction entirely offline without API keys or rate limits using an RTX GPU:

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull the vision model:
   ```bash
   ollama pull qwen2.5-vl:7b
   ```
3. Start the local server:
   ```bash
   ollama serve
   ```
4. Run the extractor:
   ```bash
   python extractors/extract_latex_local.py --limit 10 --subject Physics
   ```

### 5. Launch the Question Viewer

Browse questions, check LaTeX math formatting, inspect observation tables, and review cropped diagrams:

```bash
python viewer/preview_server.py
```

The server opens `http://localhost:8000/viewer/question_viewer.html` in your default browser.

### 6. NTA Portal Question Scraper

Download fresh question papers directly from the NTA mock test portal:

```bash
python tools/run_extractor.py --exam JEE_MAIN --limit 5
```

---

## Diagram Extraction Pipeline

The diagram cropping system uses deterministic layout analysis:

1. **Semantic Verification**: Evaluates the question statement for visual cues (`figure`, `circuit`, `diagram`, `reaction`, `shown below`, `structure`). Questions without visual references bypass image cropping.
2. **Text Boundary Detection**: Locates the `Options :` header line to establish an upper bound and exclude option regions from the main question crop.
3. **Ink Projection Profiling**: Computes horizontal ink projection arrays across the image. High-density uniform bands are identified and filtered out as text paragraphs.
4. **Structural Cluster Isolation**: Identifies isolated contours and components corresponding to diagrams, graphs, and molecular structures.
5. **Cut-to-Cut Bounding Box Calculation**: Crops tightly against the outer boundary of diagram ink strokes.
6. **2.5x Lanczos Upscaling & Anti-aliasing**: Resizes diagram crops using Lanczos4 interpolation, applies subtle Gaussian anti-aliasing, and generates crisp white strokes on transparent backgrounds.

---

## Output JSON Schema

All extractors produce standardized question records with the following 26 fields:

```json
{
  "paperId": "133",
  "paperTitle": "Paper 1 09-09-2020 Morning",
  "targetExam": "JEE_MAIN",
  "subject": "Physics",
  "questionNumber": 14,
  "topic": "Current Electricity",
  "imageUrl": "https://nta.ac.in/Uploads/Question/1/133/14/14_20191003150620.png",
  "localImagePath": "downloads/question_images/14_20191003150620.JPG",
  "type": "SINGLE_CHOICE",
  "options": [
    {"key": "1", "text": "Option 1"},
    {"key": "2", "text": "Option 2"},
    {"key": "3", "text": "Option 3"},
    {"key": "4", "text": "Option 4"}
  ],
  "correctOptions": ["2"],
  "marks": 4.0,
  "negativeMarks": 1.0,
  "subTopic": "Potentiometer Internal Resistance",
  "difficulty": "Medium",
  "exam": "JEE_MAIN",
  "latexQuestion": "In a meter bridge experiment, the null point is obtained at a distance of $l_1 = 40\\text{ cm}$...",
  "hasTable": false,
  "tableHtml": null,
  "latexOptions": [
    {"key": "1", "latex": "$2.5\\,\\Omega$", "hasDiagram": false},
    {"key": "2", "latex": "$3.0\\,\\Omega$", "hasDiagram": false},
    {"key": "3", "latex": "$4.5\\,\\Omega$", "hasDiagram": false},
    {"key": "4", "latex": "$5.0\\,\\Omega$", "hasDiagram": false}
  ],
  "hasDiagram": true,
  "diagramPath": "downloads/diagrams/Physics/diagram_14_20191003150620.png",
  "diagramBBox": [120, 45, 680, 410],
  "optionsHaveDiagrams": false,
  "optionDiagrams": {},
  "extractedVia": "gemini-3.5-flash-lite"
}
```

---

## Quota and Rate Limit Management

When using the Google Gemini Vision API:
- The primary model is configured to `gemini-3.5-flash-lite` for high free-tier daily throughput.
- If rate limits occur, the engine falls back sequentially to `gemini-3.1-flash-lite`, `gemini-flash-latest`, and `gemini-2.5-flash`.
- In case of automated copyright/recitation filter triggers (`finishReason=RECITATION`), the system applies prompt smoothing and fallback handling to preserve data continuity without stopping batch runs.

---

## License

MIT License. Copyright (c) Rajat Kolhapure.

