---
name: testing
description: Test execution, verification procedures, and validation commands for this codebase.
---

# Testing and Verification Guide

## Testing Setup Overview

This repository does not use automated test runners (such as `pytest`, `unittest` discovery suites, or `tox`), nor does it have linting or type-checking configurations (such as `flake8`, `pylint`, `mypy`, or `black`). 

Testing in this codebase is performed through targeted test harnesses, CLI dry-runs, schema validation scripts, and browser preview checks.

## Available Test Scripts

### 1. Single Image LaTeX & Diagram Extraction Test
The primary test harness is located at `tests/test_latex_extractor.py`. It tests Gemini Vision API integration, LaTeX mathematical OCR transcription, table transcription, and deterministic OpenCV cut-to-cut diagram cropping on a single image.

Command:
```powershell
python tests/test_latex_extractor.py --image downloads/question_images/<IMAGE_NAME> --gemini-key "<API_KEY>"
```

Optional Flags:
- `--stroke-color`: Stroke color hex (default: `#ffffff`)
- `--bg-mode`: Background mode (`transparent`, `white`, `dark`)

Verification Checklist for Single Image Test:
- Terminal prints question text with LaTeX `$..$` expressions.
- Options 1 to 4 are correctly extracted and printed.
- Diagram extraction prints `Has Question Diagram: True/False`.
- If true, exact boundary coordinates `[x1, y1, x2, y2]` are printed and the cropped diagram is saved to `downloads/diagrams/<Subject>/`.

## Verification Procedures for Code Changes

### 1. Python Syntax & Compilation Verification
Before committing any Python changes, verify syntax compilation across all modified files:

```powershell
python -m py_compile extractors/batch_latex_extractor.py
python -m py_compile extractors/pdf_latex_extractor.py
python -m py_compile extractors/extract_latex_local.py
python -m py_compile tools/sort_by_subject.py
python -m py_compile viewer/preview_server.py
```

### 2. CLI Interface Verification
Confirm that argument parsers, default paths, and flags execute cleanly:

```powershell
python extractors/batch_latex_extractor.py --help
python extractors/pdf_latex_extractor.py --help
python extractors/extract_latex_local.py --help
python tools/run_extractor.py --help
python tests/test_latex_extractor.py --help
```

### 3. Limited Execution (Smoke Test)
When modifying extraction logic, execute on a single question or page before processing full datasets:

- **Batch Image Extractor (Limit 1)**:
  ```powershell
  python extractors/batch_latex_extractor.py --subject Physics --input-json physics_questions.json --limit 1 --gemini-key "<API_KEY>"
  ```
- **PDF Extractor (Single Page)**:
  ```powershell
  python extractors/pdf_latex_extractor.py --pdf-dir cet --start-page 1 --end-page 1 --gemini-key "<API_KEY>"
  ```

### 4. Dataset Schema & Invariant Checks
Verify that output JSON files strictly adhere to the 26-field standard schema:

```powershell
python -c "
import json
files = ['data/extracted_physics.json', 'data/extracted_chemistry.json', 'data/extracted_mathematics.json']
for fn in files:
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data:
        keys = list(data[0].keys())
        all_match = all(list(r.keys()) == keys for r in data)
        print(f'{fn}: {len(data)} records, 26 keys={len(keys)==26}, all_consistent={all_match}')
"
```

Expected Output:
`26 keys=True`, `all_consistent=True` for each dataset.

### 5. Frontend & Viewer Verification
When modifying `viewer/question_viewer.html` or `viewer/preview_server.py`:
1. Start the server:
   ```powershell
   python viewer/preview_server.py
   ```
2. Verify HTTP 200 responses on key endpoints:
   - Root URL: `http://localhost:8000/viewer/question_viewer.html`
   - Dataset files: `http://localhost:8000/data/extracted_physics.json`
   - Diagrams: `http://localhost:8000/downloads/diagrams/Physics/<diagram_filename>.png`
3. Inspect the browser console for JavaScript exceptions, KaTeX rendering errors, or broken image URLs.
