"""
Step 2: Re-extract 153_4, 311_7, and 143_2 via Gemini API
Usage: python tools/step2_reextract.py --key YOUR_GEMINI_API_KEY
"""
import json, os, base64, re, sys, time, argparse
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DIAGRAM_DIR = PROJECT_ROOT / "downloads" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
args = parser.parse_args()
API_KEY = args.key

if not API_KEY:
    print("ERROR: Provide --key YOUR_GEMINI_API_KEY or set GEMINI_API_KEY env var")
    sys.exit(1)

RE_EXTRACT_TARGETS = [
    {"paperId": "153",   "questionNumber": 4,  "subject": "Physics",   "file": "extracted_physics.json"},
    {"paperId": "311",   "questionNumber": 7,  "subject": "Physics",   "file": "extracted_physics.json"},
    {"paperId": "143",   "questionNumber": 2,  "subject": "Chemistry", "file": "extracted_chemistry.json"},
]

PROMPT = """You are an expert LaTeX OCR and Exam Digitization engine.
Extract the question text, all 4 options, subject, topic, subtopic, and metadata from the provided NTA exam question image.

Rules:
1. Wrap all math, variables, symbols in LaTeX: inline $...$, block $$...$$
2. Transcribe Greek symbols, sub/superscripts precisely.
3. Classify: "subject" (Physics/Chemistry/Mathematics), "topic", "subTopic", "difficulty" (Easy/Medium/Hard), "exam" (JEE_MAIN/MHT_CET)
4. If diagram/circuit/graph exists: "hasDiagram": true, "diagramBoundingBox": [ymin, xmin, ymax, xmax] (0-1000 normalized)
5. Output strictly valid JSON:
{
  "subject": "...", "topic": "...", "subTopic": "...", "difficulty": "...", "exam": "...",
  "hasTable": false, "tableHtml": null,
  "hasDiagram": false, "diagramBoundingBox": null, "optionsHaveDiagrams": false,
  "latexQuestion": "...",
  "latexOptions": [
    {"key": "1", "latex": "...", "hasDiagram": false, "bbox": null},
    {"key": "2", "latex": "...", "hasDiagram": false, "bbox": null},
    {"key": "3", "latex": "...", "hasDiagram": false, "bbox": null},
    {"key": "4", "latex": "...", "hasDiagram": false, "bbox": null}
  ]
}"""


def safe_json_loads(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def call_gemini(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    for model in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": mime, "data": img_b64}}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.3}
        }
        try:
            resp = requests.post(url, json=payload, timeout=45)
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                if not candidates:
                    print(f"  [!] {model}: empty candidates"); continue
                content = candidates[0].get("content")
                if not content or "parts" not in content:
                    print(f"  [!] {model}: finishReason={candidates[0].get('finishReason')}"); continue
                parts = content["parts"]
                if not parts or "text" not in parts[0]:
                    print(f"  [!] {model}: no text"); continue
                data = safe_json_loads(parts[0]["text"])
                data["_used_model"] = model
                print(f"  OK via {model}")
                return data
            elif resp.status_code == 429:
                print(f"  [!] {model}: 429 quota. Waiting 20s..."); time.sleep(20); continue
            else:
                print(f"  [!] {model}: HTTP {resp.status_code}"); continue
        except Exception as e:
            print(f"  [!] {model}: {e}"); continue
    return {}


print("=" * 60)
print("Re-extracting via Gemini API")
print("=" * 60)

for target in RE_EXTRACT_TARGETS:
    pid = str(target["paperId"])
    qnum = target["questionNumber"]
    subject = target["subject"]
    dataset_file = target["file"]
    key = f"{pid}_{qnum}"

    print(f"\n[{key}] Q{qnum} paper {pid} ({subject})...")

    path = DATA_DIR / dataset_file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_idx = next((i for i, x in enumerate(data) if str(x.get("paperId")) == pid and x.get("questionNumber") == qnum), None)
    if q_idx is None:
        print(f"  [!] Not found in {dataset_file}"); continue
    q = data[q_idx]

    img_path = q.get("localImagePath") or q.get("imageUrl") or ""
    img_resolved = None
    for candidate in [img_path, str(PROJECT_ROOT / img_path), str(PROJECT_ROOT / img_path.lstrip("./\\"))]:
        if candidate and os.path.exists(candidate):
            img_resolved = candidate; break
    if not img_resolved:
        print(f"  [!] Image not found: {img_path}"); continue
    print(f"  Image: {img_resolved}")

    result = call_gemini(img_resolved)
    if not result or not result.get("latexQuestion"):
        print(f"  [!] Empty result — skipping"); continue

    # Preserve identity fields
    for field in ["paperId", "questionNumber", "subject", "marks", "negativeMarks",
                   "correctOptions", "type", "localImagePath", "imageUrl", "exam", "targetExam"]:
        if field in q and field not in result:
            result[field] = q[field]

    result["extractedVia"] = result.pop("_used_model", "gemini-re-extraction")
    result.pop("_recitation_filtered", None)

    # Crop diagram if newly detected
    bbox = result.get("diagramBoundingBox")
    if result.get("hasDiagram") and bbox and len(bbox) == 4:
        try:
            from PIL import Image as PILImage
            im = PILImage.open(img_resolved)
            w, h = im.size
            ymin, xmin, ymax, xmax = bbox
            left, top = int(xmin/1000*w), int(ymin/1000*h)
            right, bottom = int(xmax/1000*w), int(ymax/1000*h)
            if right > left and bottom > top:
                diag_name = f"diagram_{qnum}_{pid}.png"
                diag_path = DIAGRAM_DIR / subject / diag_name
                diag_path.parent.mkdir(parents=True, exist_ok=True)
                im.crop((left, top, right, bottom)).save(str(diag_path))
                result["diagramPath"] = f"downloads/diagrams/{subject}/{diag_name}"
                print(f"  Diagram cropped -> {result['diagramPath']}")
        except Exception as e:
            print(f"  [!] Diagram crop failed: {e}")
    elif "diagramPath" in q:
        result["diagramPath"] = q["diagramPath"]

    data[q_idx] = {**q, **result}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {dataset_file}")

    # Reset verification to pending
    verif_path = DATA_DIR / "verifications.json"
    if verif_path.exists():
        with open(verif_path, "r", encoding="utf-8") as f:
            verifs = json.load(f)
        if key in verifs:
            verifs[key]["status"] = "pending"
            verifs[key]["reason"] = ""
            with open(verif_path, "w", encoding="utf-8") as f:
                json.dump(verifs, f, indent=2, ensure_ascii=False)
            print(f"  Verification reset to pending")

    time.sleep(2)

print()
print("Done!")
