---
name: debugging
description: Systematic debugging workflow, error diagnosis, and known failure patterns in this codebase.
---

# Debugging Workflow & Troubleshooting Guide

## Standard Debugging Workflow

Always follow this 6-step discipline when resolving bugs:

```
1. Reproduce
   └── Execute the minimal command/input that consistently triggers the defect.

2. Investigate
   └── Inspect full exception tracebacks, API response payloads, and intermediate state.

3. Identify Root Cause
   └── Trace the issue to the precise line, encoding mismatch, boundary condition, or API contract.

4. Make Minimal Fix
   └── Apply the most direct, isolated fix without refactoring surrounding code or altering schemas.

5. Test
   └── Run the exact failing case to prove the defect is resolved.

6. Regression Check
   └── Validate unaffected pipelines, schema invariants (26 fields), and resume checkpoints.
```

---

## Known Failure Modes & Diagnostic Procedures

### 1. LaTeX Backslash JSON Decode Errors
- **Symptom**: `json.decoder.JSONDecodeError: Invalid \escape: line X column Y` when parsing Gemini or Ollama responses.
- **Root Cause**: The model outputs LaTeX math expressions inside JSON strings containing backslashes (e.g., `\frac`, `\tan`, `\theta`, `\alpha`). Standard JSON decoders reject `\t` as a tab or `\f` as form-feed if not properly double-escaped, while naive regex `re.sub(r"\\", r"\\\\")` corrupts existing escaped quotes (`\"`).
- **Fix**: Use the character-by-character scanner `fix_latex_json(raw_text)` implemented in `extractors/batch_latex_extractor.py` and `extractors/pdf_latex_extractor.py`. It tracks string boundaries and only doubles backslashes that do not precede valid JSON escape characters (`"`, `\`, `/`, `b`, `f`, `n`, `r`, `t`, `u`).

### 2. Gemini API Recitation Filter Blocks
- **Symptom**: `[!] Model gemini-3.5-flash-lite: no content/parts in response (finishReason=RECITATION)`.
- **Root Cause**: Google Gemini API safety filter detects verbatim matching of copyrighted NTA or coaching material and blocks text generation.
- **Fix**:
  - The extractors implement a multi-model fallback chain (`gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> `gemini-flash-latest` -> `gemini-2.5-flash`).
  - If all models are blocked by recitation, the engine logs a notice, preserves the original scanned image path, marks `extractedVia: "filtered_fallback"`, and continues to the next question rather than terminating the entire batch run.

### 3. API Rate Limits & Quota Exhaustion (HTTP 429)
- **Symptom**: `google.api_core.exceptions.ResourceExhausted: 429 Resource has been exhausted (e.g. check quota)`.
- **Root Cause**: Free tier daily or per-minute request limits reached. For example, `gemini-2.5-flash-lite` has a very low limit (20 req/day), whereas `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` offer substantially higher quotas.
- **Fix**:
  - Keep `gemini-3.5-flash-lite` as the primary model.
  - Rely on stateful resumption: all extractors write progress immediately to `data/extracted_<subject>.json`. Restarting the command automatically skips all previously extracted questions.

### 4. OpenCV Diagram Cropping Boundary Errors
- **Symptom**: Zero-dimension image crash (`cv2.error: (-215:Assertion failed) !ssize.empty()`), or diagram crop containing question text.
- **Root Cause**:
  - Text paragraph mistaken for diagram due to low ink variance.
  - Question has no diagram, but visual keyword matched (e.g., "figure of merit").
  - Inverted or negative coordinates from normalized bounding boxes (`x2 <= x1` or `y2 <= y1`).
- **Fix**:
  - Verify boundary dimensions before slicing: `if (y2 - y1) > 20 and (x2 - x1) > 20:`.
  - Check the `Options :` header delimiter to establish an upper crop boundary.
  - Use `cv2.imwrite` with valid, non-empty numpy arrays only.

### 5. Path Resolution Discrepancies
- **Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/physics_questions.json'` or `downloads/...`.
- **Root Cause**: Working directory mismatch when executing scripts from repository root (`python extractors/batch_latex_extractor.py`) versus within `extractors/` (`cd extractors; python batch_latex_extractor.py`).
- **Fix**: Resolve all paths dynamically against `PROJECT_ROOT`:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.name == "extractors") else Path(__file__).resolve().parent
  DATA_DIR = PROJECT_ROOT / "data"
  DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
  ```

### 6. Broken Image or Diagram Display in Viewer
- **Symptom**: Diagram images or scanned question images fail to load in `viewer/question_viewer.html`.
- **Root Cause**:
  - File URL containing absolute `file:///` paths that browser security policies block from HTTP contexts.
  - Relative URLs evaluated against `/viewer/` instead of server root `/`.
- **Fix**:
  - Ensure all image paths in `question_viewer.html` pass through `resolveMediaSrc(path)`.
  - Ensure `viewer/preview_server.py` serves the repository root (`DIRECTORY = REPO_ROOT`), making `/downloads/...` accessible directly.
