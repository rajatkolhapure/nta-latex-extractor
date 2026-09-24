# CLAUDE.md

## Project Overview

This repository is an automated pipeline for digitizing NTA (JEE, NEET) and MHT-CET exam question papers. It ingests question images and PDF papers, digitizes mathematical formulas into LaTeX, extracts observation tables as HTML, crops visual diagrams with deterministic OpenCV layout analysis, and outputs structured JSON datasets viewable in a browser-based KaTeX explorer.

## Technology Stack

- **Runtime**: Python 3.10+ on Windows PowerShell.
- **Core Libraries**: OpenCV (`cv2`), NumPy, PyMuPDF (`fitz`), Pillow (`PIL`), Requests, BeautifulSoup4 (`bs4`), Pydantic.
- **Scraping Engine**: Requests HTTP session simulation; optional Playwright Chromium (`playwright.sync_api`).
- **Vision APIs & Models**:
  - Primary Cloud: Google Gemini Vision API (`gemini-3.5-flash-lite`, fallback chain: `gemini-3.1-flash-lite`, `gemini-flash-latest`, `gemini-2.5-flash`).
  - Local GPU: Ollama running `qwen2.5-vl:7b` via REST API (`http://localhost:11434/api/generate`).
- **Frontend / Viewer**: Vanilla HTML5, ES6 JavaScript, Tailwind CSS (via CDN), KaTeX (via CDN with `auto-render` extension).
- **Web Server**: Python standard library `http.server` & `socketserver`.

## Directory Layout

- `extractors/`: Extraction scripts and modules.
  - `batch_latex_extractor.py`: High-throughput NTA image dataset processor with model fallbacks and resume tracking.
  - `pdf_latex_extractor.py`: PDF question paper extractor (MHT-CET / JEE) using PyMuPDF and Gemini Vision.
  - `extract_latex_local.py`: Offline GPU extractor using local Ollama instance.
  - `nta_extractor/`: Scraper package for the NTA mock test portal (`cli.py`, `downloader.py`, `classifier.py`, `filter.py`, `models.py`, `engines/`).
- `data/`: All input and output JSON datasets (`physics_questions.json`, `extracted_physics.json`, `all_nta_questions.json`, etc.).
- `downloads/`: Downloaded images (`question_images/`), cut-to-cut diagrams (`diagrams/<Subject>/`), and rendered PDF pages (`pdf_pages/`).
- `cet/`: Source question paper PDF files.
- `tools/`: Operational utility scripts (`run_extractor.py`, `sort_by_subject.py`).
- `viewer/`: Browser interface (`question_viewer.html`, `preview_server.py`).
- `tests/`: Test scripts (`test_latex_extractor.py`).

## Key Development Commands

- **Run PDF Extractor**:
  ```powershell
  python extractors/pdf_latex_extractor.py --pdf-dir cet --gemini-key "<API_KEY>"
  ```
- **Run Batch Image Extractor**:
  ```powershell
  python extractors/batch_latex_extractor.py --subject Physics --input-json physics_questions.json --gemini-key "<API_KEY>" --limit 0
  ```
- **Test Single Image**:
  ```powershell
  python tests/test_latex_extractor.py --image downloads/question_images/<IMAGE_NAME> --gemini-key "<API_KEY>"
  ```
- **Run NTA Portal Scraper**:
  ```powershell
  python tools/run_extractor.py --exam JEE_MAIN --limit 5
  ```
- **Split / Sort Master Dataset by Subject**:
  ```powershell
  python tools/sort_by_subject.py
  ```
- **Launch Local Viewer**:
  ```powershell
  python viewer/preview_server.py
  ```
- **Verify Python Syntax**:
  ```powershell
  python -m py_compile <path_to_file.py>
  ```

## Mandatory Engineering Rules

### 1. Code Investigation
- Always inspect existing code, schemas, and file paths before making assumptions.
- Do not guess field names or file locations. Check `models.py`, existing JSON files in `data/`, and script arguments first.

### 2. Minimal, Focused Changes
- Touch only lines directly relevant to the user request or bug.
- Do not add wrapper classes, redesign architecture, or introduce new abstractions unless explicitly requested.

### 3. Dependency Discipline
- Rely on standard library and existing dependencies in `requirements.txt`.
- Do not add new third-party packages (e.g., npm packages, pytest, heavy frameworks) without explicit user approval.

### 4. Zero Unrelated Refactoring
- Preserve existing comments, docstrings, and formatting in untouched functions.
- Never reformat whole files when editing a specific block.

### 5. Functionality and Schema Preservation
- Maintain strict 26-field parity across all extractor outputs:
  `paperId`, `paperTitle`, `targetExam`, `subject`, `questionNumber`, `topic`, `imageUrl`, `localImagePath`, `type`, `options`, `correctOptions`, `marks`, `negativeMarks`, `subTopic`, `difficulty`, `exam`, `latexQuestion`, `hasTable`, `tableHtml`, `latexOptions`, `hasDiagram`, `diagramPath`, `diagramBBox`, `optionsHaveDiagrams`, `optionDiagrams`, `extractedVia`.
- Option keys must always be strings (`"1"`, `"2"`, `"3"`, `"4"`).
- Always maintain stateful resumption: check previously extracted records to avoid redundant API calls.

### 6. Testing and Verification
- Verify script syntax using `python -m py_compile <file>` or `--help` after edits.
- For extractor modifications, run with `--limit 1` or `--end-page 1` before full runs.
- Verify web viewer changes by serving via `preview_server.py` and checking browser console and network responses.

### 7. Diff Review
- Always inspect `git diff` before committing or concluding a task to confirm changes are minimal and clean.

### 8. Git Safety
- Large data files (`data/`, `*.json`) and binary downloads (`downloads/`, `cet/`) must remain ignored by Git.
- Never force push (`git push --force`).
- Keep commit messages concise, imperative, and descriptive.

### 9. Tone and Output Conventions
- Do not include emojis in code, commits, or documentation.
