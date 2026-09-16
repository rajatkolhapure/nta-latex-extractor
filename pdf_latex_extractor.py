"""
Production PDF LaTeX & Cut-to-Cut Diagram Extractor for NTA / CET Papers.
Renders PDF pages at high DPI using PyMuPDF, extracts all questions with LaTeX & HTML tables,
tightly crops diagrams with 2.5x HD Lanczos upscaling (pure white transparent PNGs),
and dynamically routes each question to extracted_physics.json, extracted_chemistry.json,
or extracted_mathematics.json.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
import cv2
import numpy as np
import pymupdf
import requests

DIAGRAM_DIR = Path("./downloads/diagrams")
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PDF_PAGE = """You are an expert LaTeX OCR and Exam Digitization engine specializing in MHT-CET and JEE exam papers.
The provided image is a page from an exam question paper PDF.
Extract ALL questions present on this page into a JSON array of questions.

Rules:
1. Wrap all mathematical variables, symbols, formulas, numbers with units, and equations in LaTeX:
   - Inline math: $...$
   - Block equations: $$...$$
2. Transcribe Greek symbols (\\alpha, \\beta, \\lambda, \\Omega, \\mu, \\theta, \\pi) and sub/superscripts precisely.
3. For each question:
   - "questionNumber": integer question number (e.g. 1, 2, 3...)
   - "subject": Exactly one of ["Physics", "Chemistry", "Mathematics"] based on the question content.
   - "topic": Standard syllabus chapter (e.g. "Oscillations", "Trigonometric Functions", "Rotational Dynamics", "Differentiation", "Halogen Derivatives")
   - "subTopic": Specific concept tested
   - "difficulty": Exactly one of ["Easy", "Medium", "Hard"]
   - "exam": "MHT_CET" (or "JEE_MAIN" if JEE paper)
   - "marks": 2 for Maths, 1 for Physics/Chemistry (or as specified on page)
   - "negativeMarks": 0 for CET (or 1 for JEE)
   - "hasTable": true if an observation table exists in question body, false otherwise
   - "tableHtml": If hasTable is true, full HTML <table> string with <thead>/<tbody>/<tr>/<th>/<td> tags. Wrap cell math in $...$.
   - "hasDiagram": true if the question body contains a diagram, circuit, graph, geometric figure, or chemical structure.
   - "diagramBBox": If hasDiagram is true, provide normalized coordinates [ymin, xmin, ymax, xmax] of the diagram on a 0 to 1000 scale. If false, null.
   - "optionsHaveDiagrams": true if options contain drawn diagrams/structures, false otherwise.
   - "latexQuestion": Full transcribed question statement with LaTeX math.
   - "latexOptions": Array of options [
       {"key": "1", "latex": "...", "hasDiagram": false, "bbox": null},
       {"key": "2", "latex": "...", "hasDiagram": false, "bbox": null},
       {"key": "3", "latex": "...", "hasDiagram": false, "bbox": null},
       {"key": "4", "latex": "...", "hasDiagram": false, "bbox": null}
     ]

Output strictly valid JSON matching this schema:
{
  "questions": [
    {
      "questionNumber": 1,
      "subject": "Physics",
      "topic": "...",
      "subTopic": "...",
      "difficulty": "Medium",
      "exam": "MHT_CET",
      "marks": 1,
      "negativeMarks": 0,
      "hasTable": false,
      "tableHtml": null,
      "hasDiagram": false,
      "diagramBBox": null,
      "optionsHaveDiagrams": false,
      "latexQuestion": "...",
      "latexOptions": [
        {"key": "1", "latex": "...", "hasDiagram": false, "bbox": null},
        {"key": "2", "latex": "...", "hasDiagram": false, "bbox": null},
        {"key": "3", "latex": "...", "hasDiagram": false, "bbox": null},
        {"key": "4", "latex": "...", "hasDiagram": false, "bbox": null}
      ]
    }
  ]
}"""


def hex_to_bgr(hex_str: str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def fix_latex_json(raw: str) -> str:
    """
    Scans raw JSON string and safely doubles all LaTeX backslashes inside string literals,
    preserving escaped quotes (\") and already-escaped backslashes (\\).
    """
    out = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c == '"':
            slash_count = 0
            j = i - 1
            while j >= 0 and raw[j] == '\\':
                slash_count += 1
                j -= 1
            if slash_count % 2 == 0:
                in_string = not in_string
            out.append(c)
            i += 1
        elif in_string and c == '\\':
            if i + 1 < n and raw[i + 1] in ('"', '\\'):
                out.append(c)
                out.append(raw[i + 1])
                i += 2
            else:
                out.append('\\\\')
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def safe_json_loads(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text, strict=False)
    except Exception:
        pass

    try:
        fixed = fix_latex_json(text)
        return json.loads(fixed, strict=False)
    except Exception:
        pass

    # Secondary cleanup: escape unescaped control chars and backslashes
    cleaned = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', text)
    return json.loads(cleaned, strict=False)


def process_and_save_diagram_crop(
    crop_img: np.ndarray,
    save_path: str,
    stroke_color: str = "#ffffff",
    bg_mode: str = "transparent"
):
    """Refines bounding box, applies 2.5x Lanczos upscaling, and saves pure white transparent PNG."""
    ch, cw = crop_img.shape[:2]
    if ch < 15 or cw < 15:
        return False

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    pts = np.argwhere(thresh > 0)
    if len(pts) < 30:
        return False

    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)

    y1 = max(0, y_min - 4)
    y2 = min(ch, y_max + 5)
    x1 = max(0, x_min - 4)
    x2 = min(cw, x_max + 5)

    tight_crop_gray = gray[y1:y2, x1:x2]
    th, tw = tight_crop_gray.shape[:2]
    if th < 15 or tw < 15:
        return False

    ink_mask = np.clip((245.0 - tight_crop_gray.astype(float)) / 160.0, 0.0, 1.0)

    scale = 2.5
    up_w = int(tw * scale)
    up_h = int(th * scale)
    up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
    smooth_ink = cv2.GaussianBlur(up_ink, (3, 3), 0.5)
    final_ink = np.clip((smooth_ink - 0.06) / 0.88, 0.0, 1.0)

    stroke_bgr = hex_to_bgr(stroke_color)

    if bg_mode == "transparent":
        result = np.zeros((up_h, up_w, 4), dtype=np.uint8)
        result[:, :, 0] = stroke_bgr[0]
        result[:, :, 1] = stroke_bgr[1]
        result[:, :, 2] = stroke_bgr[2]
        result[:, :, 3] = (final_ink * 255).astype(np.uint8)
    else:
        result = np.full((up_h, up_w, 3), 255, dtype=np.uint8)
        for c in range(3):
            result[:, :, c] = (255 * (1 - final_ink) + stroke_bgr[c] * final_ink).astype(np.uint8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, result)
    return True


def call_gemini_page(page_img_b64: str, api_key: str, model: str = "gemini-3.5-flash-lite") -> list:
    models_to_try = [
        model,
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ]
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    last_err = None
    for cur_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT_PDF_PAGE},
                        {"inline_data": {"mime_type": "image/jpeg", "data": page_img_b64}}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.3
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=45)
            if resp.status_code == 200:
                rj = resp.json()
                candidates = rj.get("candidates", [])
                if not candidates:
                    last_err = f"Model {cur_model}: empty candidates"
                    continue
                content = candidates[0].get("content")
                if not content or "parts" not in content:
                    finish = candidates[0].get("finishReason", "UNKNOWN")
                    last_err = f"Model {cur_model}: no parts (finishReason={finish})"
                    continue
                parts = content["parts"]
                if not parts or "text" not in parts[0]:
                    last_err = f"Model {cur_model}: empty text"
                    continue
                try:
                    data = safe_json_loads(parts[0]["text"].strip())
                    questions = data.get("questions", [])
                    for q in questions:
                        q["_used_model"] = cur_model
                    return questions
                except Exception as parse_err:
                    last_err = f"Model {cur_model}: JSON parse error: {parse_err}"
                    print(f"  [!] {last_err}")
                    continue
            elif resp.status_code == 429:
                last_err = f"Model {cur_model} 429 Quota Exceeded"
                continue
            else:
                last_err = f"API Error {resp.status_code}: {resp.text[:150]}"
                continue
        except requests.exceptions.RequestException as e:
            last_err = f"Model {cur_model} network error: {e}"
            continue

    print(f"  [!] All models exhausted on page. Last error: {last_err}")
    return []


def process_pdf_file(
    pdf_path: str,
    api_key: str,
    subject_data: dict,
    subject_files: dict,
    processed_keys: set,
    stroke_color: str = "#ffffff",
    bg_mode: str = "transparent",
    dpi: int = 180,
    start_page: int = 1,
    end_page: int = 0
):
    pdf_stem = Path(pdf_path).stem
    print(f"\n=================================================================")
    print(f" Processing PDF : {Path(pdf_path).name}")
    print(f"=================================================================")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"[!] Error opening PDF {pdf_path}: {e}")
        return 0

    total_pages = len(doc)
    actual_end = total_pages if end_page <= 0 else min(end_page, total_pages)
    total_extracted_this_pdf = 0

    for page_idx in range(start_page - 1, actual_end):
        page_num = page_idx + 1
        print(f"\n--- Page {page_num}/{total_pages} ---", flush=True)

        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n >= 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        h_px, w_px = img_bgr.shape[:2]

        page_img_dir = Path("./downloads/pdf_pages") / pdf_stem
        page_img_dir.mkdir(parents=True, exist_ok=True)
        page_img_path = page_img_dir / f"page_{page_num}.png"
        cv2.imwrite(str(page_img_path), img_bgr)

        _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        questions = call_gemini_page(img_b64, api_key)
        if not questions:
            print(f"  [-] No questions extracted on page {page_num}")
            continue

        print(f"  [+] Extracted {len(questions)} question(s) from page {page_num}:")

        for q in questions:
            q_num = q.get("questionNumber")
            dedupe_key = f"{pdf_stem}_p{page_num}_q{q_num}"
            if dedupe_key in processed_keys or f"{pdf_stem}_q{q_num}" in processed_keys:
                print(f"    [-] Skipping Q{q_num} (already extracted)")
                continue

            subj = (q.get("subject") or "Physics").capitalize()
            if subj not in subject_data:
                subj = "Physics"

            target_file = subject_files[subj]
            diag_dir = DIAGRAM_DIR / subj
            diag_dir.mkdir(parents=True, exist_ok=True)

            has_diag = bool(q.get("hasDiagram", False))
            diag_bbox = q.get("diagramBBox")
            diag_saved_path = None

            if has_diag and diag_bbox and len(diag_bbox) == 4:
                ymin, xmin, ymax, xmax = [float(c) / 1000.0 for c in diag_bbox]
                y1 = max(0, int(ymin * h_px))
                y2 = min(h_px, int(ymax * h_px))
                x1 = max(0, int(xmin * w_px))
                x2 = min(w_px, int(xmax * w_px))

                if (y2 - y1) > 20 and (x2 - x1) > 20:
                    raw_diag_crop = img_bgr[y1:y2, x1:x2]
                    diag_filename = f"diagram_{pdf_stem}_p{page_num}_q{q_num}.png"
                    diag_save_full = str(diag_dir / diag_filename)
                    if process_and_save_diagram_crop(raw_diag_crop, diag_save_full, stroke_color, bg_mode):
                        diag_saved_path = diag_save_full

            key_map = {"A": "1", "B": "2", "C": "3", "D": "4", "1": "1", "2": "2", "3": "3", "4": "4"}

            opts_have_diag = bool(q.get("optionsHaveDiagrams", False))
            option_diagrams = {}
            if opts_have_diag:
                for opt in q.get("latexOptions", []):
                    raw_opt_key = str(opt.get("key", "")).strip().upper()
                    opt_key = key_map.get(raw_opt_key, raw_opt_key)
                    opt_bbox = opt.get("bbox")
                    if opt.get("hasDiagram") and opt_bbox and len(opt_bbox) == 4:
                        oymin, oxmin, oymax, oxmax = [float(c) / 1000.0 for c in opt_bbox]
                        oy1 = max(0, int(oymin * h_px))
                        oy2 = min(h_px, int(oymax * h_px))
                        ox1 = max(0, int(oxmin * w_px))
                        ox2 = min(w_px, int(oxmax * w_px))
                        if (oy2 - oy1) > 15 and (ox2 - ox1) > 15:
                            opt_crop = img_bgr[oy1:oy2, ox1:ox2]
                            opt_filename = f"option_{opt_key}_{pdf_stem}_p{page_num}_q{q_num}.png"
                            opt_save_full = str(diag_dir / opt_filename)
                            if process_and_save_diagram_crop(opt_crop, opt_save_full, stroke_color, bg_mode):
                                option_diagrams[opt_key] = opt_save_full

            clean_options = []
            clean_latex_options = []
            for opt in q.get("latexOptions", []):
                raw_k = str(opt.get("key", "")).strip().upper()
                k = key_map.get(raw_k, raw_k)
                txt = opt.get("latex") or opt.get("text") or ""
                clean_options.append({"key": k, "text": f"Option {k}"})
                clean_latex_options.append({
                    "key": k,
                    "latex": txt,
                    "hasDiagram": bool(opt.get("hasDiagram", False))
                })

            if not clean_options:
                clean_options = [{"key": str(i), "text": f"Option {i}"} for i in range(1, 5)]

            exam_val = q.get("exam") or "MHT_CET"
            if "CET" in pdf_stem.upper() or "MHCET" in pdf_stem.upper():
                exam_val = "MHT_CET"

            # Exact schema matching batch_latex_extractor.py
            record = {
                "paperId": pdf_stem,
                "paperTitle": pdf_stem.replace("_", " "),
                "targetExam": exam_val,
                "subject": subj,
                "questionNumber": q_num,
                "topic": q.get("topic") or "General",
                "imageUrl": f"file:///{page_img_path.resolve().as_posix()}",
                "localImagePath": str(page_img_path),
                "type": "SINGLE_CHOICE",
                "options": clean_options,
                "correctOptions": [],
                "marks": float(q.get("marks") or (2.0 if subj == "Mathematics" else 1.0)),
                "negativeMarks": float(q.get("negativeMarks") or 0.0),
                "subTopic": q.get("subTopic", ""),
                "difficulty": q.get("difficulty") or "Medium",
                "exam": exam_val,
                "latexQuestion": q.get("latexQuestion", ""),
                "hasTable": bool(q.get("hasTable", False)),
                "tableHtml": q.get("tableHtml") or None,
                "latexOptions": clean_latex_options,
                "hasDiagram": bool(diag_saved_path),
                "diagramPath": diag_saved_path,
                "diagramBBox": [int(x1), int(y1), int(x2), int(y2)] if (has_diag and diag_saved_path) else None,
                "optionsHaveDiagrams": bool(option_diagrams),
                "optionDiagrams": option_diagrams,
                "extractedVia": q.get("_used_model", "gemini-3.5-flash-lite")
            }

            subject_data[subj].append(record)
            processed_keys.add(dedupe_key)
            processed_keys.add(f"{pdf_stem}_q{q_num}")
            total_extracted_this_pdf += 1

            status_note = f"    [*] Q{q_num}: {subj} > {record['topic']} [{record['difficulty']}]"
            if diag_saved_path:
                status_note += f" [HD Diagram Saved]"
            if record["hasTable"]:
                status_note += f" [TABLE]"
            print(status_note)

        for s_name, s_file in subject_files.items():
            if subject_data[s_name]:
                with open(s_file, "w", encoding="utf-8") as f:
                    json.dump(subject_data[s_name], f, indent=2, ensure_ascii=False)

        time.sleep(3.0)

    doc.close()
    return total_extracted_this_pdf


def main():
    parser = argparse.ArgumentParser(description="PDF Exam Question & Cut-to-Cut Diagram Extractor")
    parser.add_argument("--pdf", default=None, help="Path to a single PDF file")
    parser.add_argument("--pdf-dir", default="cet", help="Directory containing PDF files (default: cet)")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"), help="API Key")
    parser.add_argument("--stroke-color", default="#ffffff", help="Diagram stroke color (default: #ffffff)")
    parser.add_argument("--bg-mode", default="transparent", choices=["transparent", "white", "dark"])
    parser.add_argument("--dpi", type=int, default=180, help="Page rendering DPI (default: 180)")
    parser.add_argument("--start-page", type=int, default=1, help="Start page (default: 1)")
    parser.add_argument("--end-page", type=int, default=0, help="End page (0 = all)")

    args = parser.parse_args()

    if not args.gemini_key:
        print("\n[!] Please pass --gemini-key YOUR_API_KEY")
        return

    subject_files = {
        "Physics": "extracted_physics.json",
        "Chemistry": "extracted_chemistry.json",
        "Mathematics": "extracted_mathematics.json",
    }

    subject_data = {"Physics": [], "Chemistry": [], "Mathematics": []}
    processed_keys = set()

    for s_name, s_file in subject_files.items():
        if os.path.exists(s_file):
            try:
                with open(s_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    subject_data[s_name] = items
                    for item in items:
                        pdf_src = item.get("paperId", "")
                        q_num = item.get("questionNumber", "")
                        if pdf_src and q_num:
                            processed_keys.add(f"{pdf_src}_q{q_num}")
                            page_m = re.search(r"page_(\d+)", str(item.get("localImagePath", "")))
                            if page_m:
                                processed_keys.add(f"{pdf_src}_p{page_m.group(1)}_q{q_num}")
            except Exception:
                pass

    print("=================================================================")
    print(" Production PDF LaTeX & Cut-to-Cut Diagram Extractor")
    print(f" Existing Questions Loaded : Physics={len(subject_data['Physics'])}, Chemistry={len(subject_data['Chemistry'])}, Mathematics={len(subject_data['Mathematics'])}")
    print(f" Stroke Color              : {args.stroke_color}")
    print(f" Background Mode           : {args.bg_mode}")
    print("=================================================================")

    pdf_files_to_run = []
    if args.pdf:
        if os.path.exists(args.pdf):
            pdf_files_to_run.append(args.pdf)
        else:
            print(f"[!] PDF not found: {args.pdf}")
            return
    else:
        target_dir = args.pdf_dir
        if not os.path.exists(target_dir):
            alt_dir = r"C:\Users\rmk19\Downloads\cet"
            if os.path.exists(alt_dir):
                target_dir = alt_dir

        if not os.path.exists(target_dir):
            print(f"[!] PDF directory not found: {args.pdf_dir}")
            return

        for root, _, files in os.walk(target_dir):
            for f in files:
                if f.lower().endswith(".pdf") and "solution" not in f.lower() and "answer_key" not in f.lower():
                    pdf_files_to_run.append(os.path.join(root, f))

    print(f"Found {len(pdf_files_to_run)} question paper PDF(s) to process.")

    grand_total = 0
    for idx, p_file in enumerate(pdf_files_to_run, 1):
        print(f"\n[{idx}/{len(pdf_files_to_run)}] Processing: {os.path.basename(p_file)}")
        count = process_pdf_file(
            pdf_path=p_file,
            api_key=args.gemini_key,
            subject_data=subject_data,
            subject_files=subject_files,
            processed_keys=processed_keys,
            stroke_color=args.stroke_color,
            bg_mode=args.bg_mode,
            dpi=args.dpi,
            start_page=args.start_page,
            end_page=args.end_page
        )
        grand_total += count

    print("\n=================================================================")
    print(f" [OK] Complete! Extracted {grand_total} question(s) across all PDFs.")
    print(f" Physics      : {len(subject_data['Physics'])} questions in extracted_physics.json")
    print(f" Chemistry    : {len(subject_data['Chemistry'])} questions in extracted_chemistry.json")
    print(f" Mathematics  : {len(subject_data['Mathematics'])} questions in extracted_mathematics.json")
    print("=================================================================")


if __name__ == "__main__":
    main()
