import http.server
import socketserver
import webbrowser
import json
import os
import sys
from pathlib import Path

# Import crop_option_regions from the extractors package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractors"))
try:
    from batch_latex_extractor import crop_option_regions
    _CROP_AVAILABLE = True
except Exception:
    _CROP_AVAILABLE = False

PORT = 8000
REPO_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.name == "viewer") else Path(__file__).resolve().parent
DIRECTORY = REPO_ROOT
VERIFICATIONS_FILE = REPO_ROOT / "data" / "verifications.json"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/question_viewer.html"):
            self.send_response(302)
            self.send_header("Location", "/viewer/question_viewer.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/verify":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                paper_id = str(data.get('paperId'))
                q_num = str(data.get('questionNumber'))
                status = data.get('status')
                diag_path = data.get('diagramPath')
                reason = data.get('reason')

                # Load existing
                verifications = {}
                if VERIFICATIONS_FILE.exists():
                    try:
                        with open(VERIFICATIONS_FILE, 'r', encoding='utf-8') as f:
                            verifications = json.load(f)
                    except: pass

                key = f"{paper_id}_{q_num}"
                if status == 'pending':
                    if key in verifications:
                        del verifications[key]
                else:
                    record = {
                        "status": status,
                        "diagramPath": diag_path
                    }
                    if reason:
                        record["reason"] = str(reason).strip()
                    elif key in verifications and "reason" in verifications[key] and status == verifications[key].get("status"):
                        record["reason"] = verifications[key]["reason"]
                    verifications[key] = record

                # Save
                VERIFICATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(VERIFICATIONS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(verifications, f, indent=2, ensure_ascii=False)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
            return
        elif self.path == "/api/crop-options":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                if not _CROP_AVAILABLE:
                    raise RuntimeError("crop_option_regions not available (import failed)")

                data = json.loads(post_data)
                paper_id = str(data.get('paperId', ''))
                q_num = str(data.get('questionNumber', ''))
                subject = data.get('subject', 'Physics')

                # Locate subject JSON
                subj_file = REPO_ROOT / "data" / f"extracted_{subject.lower()}.json"
                if not subj_file.exists():
                    raise FileNotFoundError(f"Subject file not found: {subj_file.name}")

                with open(subj_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)

                # Find the question record
                q_rec = next(
                    (q for q in questions
                     if str(q.get('paperId')) == paper_id
                     and str(q.get('questionNumber')) == q_num),
                    None
                )
                if q_rec is None:
                    raise ValueError(f"Question {paper_id}_{q_num} not found in {subject}")

                # Resolve local image path
                local_path = q_rec.get('localImagePath', '')
                img_path = str(REPO_ROOT / local_path.lstrip('./').lstrip('/'))

                stem = Path(img_path).stem

                # Build opts_latex — fall back to 4-way crop for filtered_fallback questions
                opts_latex = q_rec.get('latexOptions') or []
                if not opts_latex:
                    opts_latex = [
                        {"key": "1", "hasDiagram": True},
                        {"key": "2", "hasDiagram": True},
                        {"key": "3", "hasDiagram": True},
                        {"key": "4", "hasDiagram": True},
                    ]

                diag_dir = REPO_ROOT / "downloads" / "diagrams" / subject / "options"
                diag_dir.mkdir(parents=True, exist_ok=True)

                result = crop_option_regions(
                    img_path=img_path,
                    opts_latex=opts_latex,
                    diag_dir=diag_dir,
                    stem=stem,
                    bg_mode="transparent",
                    stroke_color="#ffffff"
                )

                # Write result back into the subject JSON
                q_rec["optionDiagrams"] = result
                if any(v for v in result.values()):
                    q_rec["optionsHaveDiagrams"] = True
                with open(subj_file, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=2, ensure_ascii=False)

                # Build relative forward-slash paths for the HTTP client
                rel_result = {}
                for k, v in result.items():
                    if v:
                        try:
                            rel_result[k] = str(Path(v).relative_to(REPO_ROOT)).replace("\\", "/")
                        except ValueError:
                            rel_result[k] = str(v).replace("\\", "/")
                    else:
                        rel_result[k] = None

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"optionDiagrams": rel_result}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        self.send_response(404)
        self.end_headers()

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            webbrowser.open(f"http://localhost:{PORT}/viewer/question_viewer.html")
        except: pass
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
