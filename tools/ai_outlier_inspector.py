#!/usr/bin/env python3
"""
AI Outlier & Diagram Anomaly Inspector

Audits questions using Gemini Vision with high-resolution image analysis.
Flags questions that:
  - Contain visual diagrams, graphs, schematics, circuits, or chemical structures not properly isolated
  - Have options containing visual structures needing cropping
  - Have out-of-the-ordinary layouts (e.g., composite figures, side-by-side text, multi-column options)
Routes all outlier questions into `data/manual_crop_queue.json` for rapid browser cropping.
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

INSPECTOR_SYSTEM_INSTRUCTION = """You are an expert scientific document quality auditor specializing in STEM exam paper OCR and layout analysis.
Your job is to examine an original exam question image alongside its current extracted data, and determine if this question needs MANUAL CROPPING or has any out-of-the-ordinary visual structure.

Check carefully for:
1. QUESTION DIAGRAM: Does the question contain a diagram, circuit, graph, geometric figure, apparatus setup, table, or chemical structure?
   - If YES, does the current extraction have a valid diagram crop, or is it missing/incomplete?
2. OPTION DIAGRAMS: Do options (1, 2, 3, 4) contain visual figures, structures, circuit branches, or graphs?
   - If YES, are options missing crops?
3. OUT OF THE ORDINARY: Is there any unconventional layout, such as side-by-side text and diagram, composite/multi-part diagrams, or diagrams embedded inside the text flow?
4. CORRECTION NEEDED: If the OCR or layout is flawed, unusual, or requires human visual isolation via crop, mark needs_manual_crop as true.

Output strictly valid JSON with no markdown fences, no surrounding text:
{
  "needs_manual_crop": boolean,
  "has_visual_diagram": boolean,
  "options_have_visuals": boolean,
  "anomaly_types": [string],
  "reason": string,
  "suggested_targets": ["diagram", "option_1", "option_2", "option_3", "option_4"]
}
"""

def call_gemini_vision(image_bytes: bytes, mime_type: str, extracted_json_summary: str, api_key: str, model: str = "gemini-2.5-flash"):
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = (
        f"{INSPECTOR_SYSTEM_INSTRUCTION}\n\n"
        f"CURRENT EXTRACTED DATA FOR THIS QUESTION:\n{extracted_json_summary}\n\n"
        "Evaluate the attached image now and return the JSON assessment."
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_img
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }
    
    # Try preferred model, fallback to gemini-3.5-flash-lite or gemini-2.5-flash
    models_to_try = [model, "gemini-2.5-flash", "gemini-3.5-flash-lite"]
    # Deduplicate while preserving order
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]
    
    last_err = None
    for m in models_to_try:
        url = GEMINI_API_URL.format(model=m, api_key=api_key)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                # Clean up if enclosed in markdown fences
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                return json.loads(cleaned)
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            last_err = f"HTTP {he.code} for {m}: {err_body}"
            if he.code == 404:
                continue # Try next model
            if he.code == 429:
                print("[429 Rate limit - sleeping 15s]...", end=" ", flush=True)
                time.sleep(15)
                # Retry once
                try:
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                        cleaned = text.strip()
                        if cleaned.startswith("```"):
                            lines = cleaned.splitlines()
                            if lines[0].startswith("```"):
                                lines = lines[1:]
                            if lines and lines[-1].startswith("```"):
                                lines = lines[:-1]
                            cleaned = "\n".join(lines).strip()
                        return json.loads(cleaned)
                except Exception:
                    pass
            break
        except Exception as ex:
            last_err = str(ex)
            break
            
    raise RuntimeError(f"Gemini API call failed: {last_err}")


def inspect_question(q: dict, api_key: str, model: str = "gemini-2.5-flash") -> dict:
    local_path = q.get("localImagePath", "")
    if not local_path:
        return {"needs_manual_crop": False, "reason": "No local image path"}
        
    full_path = REPO_ROOT / local_path.lstrip("./").lstrip("/")
    if not full_path.exists():
        return {"needs_manual_crop": False, "reason": f"Image file not found: {full_path}"}
        
    with open(full_path, "rb") as f:
        img_bytes = f.read()
        
    ext = full_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    
    summary = {
        "paperId": q.get("paperId"),
        "questionNumber": q.get("questionNumber"),
        "subject": q.get("subject"),
        "hasDiagram": q.get("hasDiagram", False),
        "diagramPath": q.get("diagramPath"),
        "hasTable": q.get("hasTable", False),
        "optionsHaveDiagrams": q.get("optionsHaveDiagrams", False),
        "optionDiagrams": q.get("optionDiagrams", {}),
        "latexQuestion": (q.get("latexQuestion") or "")[:300],
        "optionsCount": len(q.get("latexOptions") or q.get("options") or [])
    }
    
    return call_gemini_vision(img_bytes, mime, json.dumps(summary, indent=2), api_key, model=model)


def main():
    parser = argparse.ArgumentParser(description="Inspect questions with Gemini Vision to detect outliers & missing crops.")
    parser.add_argument("--dataset", default="data/re_extracted_flagged.json", help="Path to input dataset JSON")
    parser.add_argument("--output", default="data/manual_crop_queue.json", help="Path to output outlier queue JSON")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Gemini API Key")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini Vision model name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to process (0 = all)")
    parser.add_argument("--force-recheck", action="store_true", help="Recheck even if question already inspected")
    args = parser.parse_args()
    
    in_path = REPO_ROOT / args.dataset
    out_path = REPO_ROOT / args.output
    
    if not in_path.exists():
        print(f"Dataset {in_path} does not exist.")
        sys.exit(1)
        
    with open(in_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} questions from {args.dataset}")
    
    # Load existing queue if present
    queue = []
    queue_keys = set()
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                queue = json.load(f)
            for q in queue:
                queue_keys.add(f"{q.get('paperId')}_{q.get('questionNumber')}")
            print(f"Found existing queue with {len(queue)} questions.")
        except Exception as e:
            print(f"Warning: could not read existing queue: {e}")
            queue = []
            
    processed_count = 0
    flagged_count = 0
    
    for i, q in enumerate(questions):
        if args.limit and processed_count >= args.limit:
            break
            
        key = f"{q.get('paperId')}_{q.get('questionNumber')}"
        if key in queue_keys and not args.force_recheck:
            continue
            
        print(f"[{i+1}/{len(questions)}] Inspecting {key} ({q.get('subject')})...", end=" ", flush=True)
        try:
            report = inspect_question(q, args.api_key, model=args.model)
            processed_count += 1
            needs_crop = report.get("needs_manual_crop", False)
            
            if needs_crop:
                flagged_count += 1
                q_copy = dict(q)
                q_copy["aiAnomalyReport"] = report
                # Add to queue
                if key in queue_keys:
                    # Update in place
                    for idx, eq in enumerate(queue):
                        if f"{eq.get('paperId')}_{eq.get('questionNumber')}" == key:
                            queue[idx] = q_copy
                            break
                else:
                    queue.append(q_copy)
                    queue_keys.add(key)
                print(f"⚠️ OUTLIER: {report.get('reason', '')[:60]}...")
            else:
                print("✓ OK")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Save output queue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Inspected: {processed_count}, Outliers Added: {flagged_count}, Total in Queue: {len(queue)}")
    print(f"Queue saved to: {out_path}")

if __name__ == "__main__":
    main()
