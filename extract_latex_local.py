"""
Local VLM LaTeX & Diagram Extraction Engine for RTX 4070 Laptop GPU.
Supports Ollama (e.g. qwen2.5-vl:7b) or HuggingFace transformers with 4-bit AWQ/GPTQ.

Usage:
  python extract_latex_local.py --limit 5
"""

import argparse
import base64
import json
import os
import re
from pathlib import Path
from PIL import Image

try:
    import requests
except ImportError:
    pass

DIAGRAM_OUTPUT_DIR = Path("./downloads/diagrams")
DIAGRAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_diagram_bbox(image_path: str, bbox: list, save_path: str) -> bool:
    """
    Crops diagram from image given normalized bbox [ymin, xmin, ymax, xmax] (0 to 1000).
    """
    try:
        im = Image.open(image_path)
        w, h = im.size
        ymin, xmin, ymax, xmax = bbox
        left = int((xmin / 1000.0) * w)
        top = int((ymin / 1000.0) * h)
        right = int((xmax / 1000.0) * w)
        bottom = int((ymax / 1000.0) * h)

        # Sanity check
        if right <= left or bottom <= top:
            return False

        cropped = im.crop((left, top, right, bottom))
        cropped.save(save_path)
        return True
    except Exception as e:
        print(f"Error cropping diagram {image_path}: {e}")
        return False


def call_ollama_vlm(image_path: str, prompt: str, model: str = "qwen2.5-vl:7b") -> dict:
    """
    Queries local Ollama instance running on RTX 4070 (http://localhost:11434).
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
    }

    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
    if resp.status_code == 200:
        raw_text = resp.json().get("response", "{}")
        return json.loads(raw_text)
    else:
        raise RuntimeError(f"Ollama returned status {resp.status_code}: {resp.text}")


SYSTEM_PROMPT = """You are an expert LaTeX OCR and Exam Digitization engine.
Extract the question text and all 4 options from the provided NTA exam question image.

Rules:
1. Wrap all mathematical variables, symbols, numbers with units, and equations in LaTeX:
   - Inline math: $...$
   - Block equations: $$...$$
2. Transcribe Greek symbols (\alpha, \beta, \lambda, \Omega, \mu) and sub/superscripts precisely.
3. If the question contains a diagram, circuit, graph, or chemical structure:
   - Set "hasDiagram": true
   - Provide "diagramBoundingBox": [ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates.
   - Describe the diagram briefly in "diagramDescription".
4. Output strictly valid JSON with this schema:
{
  "hasDiagram": true/false,
  "diagramBoundingBox": [ymin, xmin, ymax, xmax] or null,
  "diagramDescription": "...",
  "latexQuestion": "...",
  "latexOptions": [
    {"key": "1", "latex": "..."},
    {"key": "2", "latex": "..."},
    {"key": "3", "latex": "..."},
    {"key": "4", "latex": "..."}
  ]
}"""


def process_dataset(
    input_json: str = "all_nta_questions.json",
    output_json: str = "all_nta_questions_latex.json",
    limit: int = None,
    subject: str = None
):
    with open(input_json, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if subject:
        questions = [q for q in questions if q.get("subject", "").lower() == subject.lower()]
        print(f"Filtered by subject '{subject}': {len(questions)} questions found.")

    if limit:
        questions = questions[:limit]

    print(f"Starting local VLM extraction for {len(questions)} questions on RTX 4070...")
    print("Checking for Ollama at http://localhost:11434...")

    for idx, q in enumerate(questions, 1):
        img_path = q.get("localImagePath")
        if not img_path or not os.path.exists(img_path):
            continue

        print(f"[{idx}/{len(questions)}] Processing Q{q['questionNumber']} ({q['paperTitle']} - {q.get('subject')})...")
        try:
            result = call_ollama_vlm(img_path, SYSTEM_PROMPT)
            q["hasDiagram"] = result.get("hasDiagram", False)
            q["latexQuestion"] = result.get("latexQuestion", "")
            q["latexOptions"] = result.get("latexOptions", [])

            bbox = result.get("diagramBoundingBox")
            if q["hasDiagram"] and bbox and len(bbox) == 4:
                diag_filename = f"{q['paperId']}_q{q['questionNumber']}_diagram.png"
                diag_save_path = DIAGRAM_OUTPUT_DIR / diag_filename
                ok = crop_diagram_bbox(img_path, bbox, str(diag_save_path))
                if ok:
                    q["diagramLocalPath"] = str(diag_save_path.as_posix())

            # Save progress
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  Note: Ollama not running or error ({e}). See guide below to start Ollama.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local VLM LaTeX & Diagram Extraction on RTX 4070")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to test (0 for all)")
    parser.add_argument("--subject", default=None, choices=["Physics", "Chemistry", "Mathematics"], help="Filter by subject (Physics, Chemistry, Mathematics)")
    parser.add_argument("--input-json", default="all_nta_questions.json", help="Input questions JSON file")
    parser.add_argument("--output-json", default="all_nta_questions_latex.json", help="Output JSON file")
    args = parser.parse_args()
    process_dataset(
        input_json=args.input_json,
        output_json=args.output_json,
        limit=args.limit if args.limit > 0 else None,
        subject=args.subject
    )
