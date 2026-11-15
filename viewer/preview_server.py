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
        elif self.path == "/api/manual-crop":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                import cv2
                import numpy as np

                data = json.loads(post_data)
                paper_id = str(data.get('paperId', ''))
                q_num = str(data.get('questionNumber', ''))
                subject = data.get('subject', 'Physics')
                raw_target = str(data.get('target', 'diagram'))
                target = raw_target.replace('option_', '') # 'diagram' or '1', '2', '3', '4'
                dataset = data.get('dataset', f'data/extracted_{subject.lower()}.json')
                
                # Bounding box in raw image pixel coordinates or ratio
                coords = data.get('coords') or {}
                bbox = data.get('bbox')
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    coords = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3],
                        'naturalWidth': data.get('naturalWidth'),
                        'naturalHeight': data.get('naturalHeight'),
                    }
                
                # Locate dataset file
                ds_file = REPO_ROOT / dataset.replace('\\', '/').lstrip('/')
                if not ds_file.exists():
                    ds_file = REPO_ROOT / "data" / f"extracted_{subject.lower()}.json"

                with open(ds_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)

                q_rec = next(
                    (q for q in questions
                     if str(q.get('paperId')) == paper_id
                     and str(q.get('questionNumber')) == q_num),
                    None
                )
                if q_rec is None:
                    # Search other datasets
                    for alt_name in [f"extracted_{subject.lower()}.json", "re_extracted_flagged.json", "manual_crop_queue.json"]:
                        alt_file = REPO_ROOT / "data" / alt_name
                        if alt_file.exists():
                            with open(alt_file, 'r', encoding='utf-8') as f:
                                alt_qs = json.load(f)
                            found = next((q for q in alt_qs if str(q.get('paperId')) == paper_id and str(q.get('questionNumber')) == q_num), None)
                            if found:
                                q_rec = found
                                ds_file = alt_file
                                questions = alt_qs
                                break
                    if q_rec is None:
                        raise ValueError(f"Question {paper_id}_{q_num} not found")

                local_path = q_rec.get('localImagePath', '')
                img_path = str(REPO_ROOT / local_path.lstrip('./').lstrip('/'))
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Source image not found: {img_path}")

                ih, iw = img.shape[:2]
                
                # Scale coordinates if natural dimensions provided
                nw = coords.get('naturalWidth') or iw
                nh = coords.get('naturalHeight') or ih
                scale_x = iw / float(nw)
                scale_y = ih / float(nh)
                
                x = int(coords.get('x', 0) * scale_x)
                y = int(coords.get('y', 0) * scale_y)
                w = int(coords.get('width', 0) * scale_x)
                h = int(coords.get('height', 0) * scale_y)

                # Clamp bounds
                x1 = max(0, min(iw - 1, x))
                y1 = max(0, min(ih - 1, y))
                x2 = max(0, min(iw, x + w))
                y2 = max(0, min(ih, y + h))

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    raise ValueError("Selection box too small")

                # Fast clean: transparent background with crisp white ink matching viewer themes
                crop = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ch, cw = crop.shape[:2]
                alpha = np.clip((245.0 - gray.astype(np.float32)) * 1.5, 0.0, 255.0).astype(np.uint8)
                out = np.zeros((ch, cw, 4), dtype=np.uint8)
                out[:, :, :3] = 255
                out[:, :, 3] = alpha

                stem = Path(img_path).stem
                if target == 'diagram':
                    diag_dir = REPO_ROOT / "downloads" / "diagrams" / subject
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    save_path = diag_dir / f"diagram_{stem}.png"
                    cv2.imwrite(str(save_path), out)
                    
                    rel_path = str(save_path.relative_to(REPO_ROOT)).replace("\\", "/")
                    q_rec["hasDiagram"] = True
                    q_rec["diagramPath"] = rel_path
                    q_rec["diagramBBox"] = [x1, y1, x2, y2]
                else:
                    opt_dir = REPO_ROOT / "downloads" / "diagrams" / subject / "options"
                    opt_dir.mkdir(parents=True, exist_ok=True)
                    save_path = opt_dir / f"option_{target}_{stem}.png"
                    cv2.imwrite(str(save_path), out)
                    
                    rel_path = str(save_path.relative_to(REPO_ROOT)).replace("\\", "/")
                    if "optionDiagrams" not in q_rec or not isinstance(q_rec["optionDiagrams"], dict):
                        q_rec["optionDiagrams"] = {}
                    q_rec["optionDiagrams"][target] = rel_path
                    q_rec["optionsHaveDiagrams"] = True

                with open(ds_file, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=2, ensure_ascii=False)

                # Sync update to any other dataset files containing this question
                for alt_name in [f"extracted_{subject.lower()}.json", "re_extracted_flagged.json", "manual_crop_queue.json"]:
                    alt_path = REPO_ROOT / "data" / alt_name
                    if alt_path.exists() and alt_path.resolve() != ds_file.resolve():
                        try:
                            with open(alt_path, 'r', encoding='utf-8') as f:
                                alt_data = json.load(f)
                            changed = False
                            for aq in alt_data:
                                if str(aq.get('paperId')) == paper_id and str(aq.get('questionNumber')) == q_num:
                                    if target == 'diagram':
                                        aq["hasDiagram"] = True
                                        aq["diagramPath"] = rel_path
                                        aq["diagramBBox"] = [x1, y1, x2, y2]
                                    else:
                                        if "optionDiagrams" not in aq or not isinstance(aq["optionDiagrams"], dict):
                                            aq["optionDiagrams"] = {}
                                        aq["optionDiagrams"][target] = rel_path
                                        aq["optionsHaveDiagrams"] = True
                                    changed = True
                            if changed:
                                with open(alt_path, 'w', encoding='utf-8') as f:
                                    json.dump(alt_data, f, indent=2, ensure_ascii=False)
                        except Exception:
                            pass

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "target": target,
                    "path": rel_path,
                    "bbox": [x1, y1, x2, y2]
                }).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        elif self.path == "/api/remove-diagram":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                paper_id = str(data.get('paperId', ''))
                q_num = str(data.get('questionNumber', ''))
                subject = data.get('subject', 'Physics')
                dataset = data.get('dataset', f'data/extracted_{subject.lower()}.json')
                target = str(data.get('target', 'diagram'))

                # Update across all candidate dataset files
                for alt_name in [dataset, f"data/extracted_{subject.lower()}.json", "data/re_extracted_flagged.json", "data/manual_crop_queue.json"]:
                    p = REPO_ROOT / alt_name.replace('\\', '/').lstrip('/')
                    if p.exists():
                        try:
                            with open(p, 'r', encoding='utf-8') as f:
                                qs = json.load(f)
                            updated = False
                            for q in qs:
                                if str(q.get('paperId')) == paper_id and str(q.get('questionNumber')) == q_num:
                                    if target == 'diagram':
                                        q['hasDiagram'] = False
                                        q['diagramPath'] = None
                                        q['diagramBBox'] = None
                                    elif target.startswith('option_'):
                                        opt_k = target.replace('option_', '')
                                        if 'optionDiagrams' in q and isinstance(q['optionDiagrams'], dict):
                                            q['optionDiagrams'].pop(opt_k, None)
                                            q['optionsHaveDiagrams'] = any(bool(v) for v in q['optionDiagrams'].values())
                                    updated = True
                            if updated:
                                with open(p, 'w', encoding='utf-8') as f:
                                    json.dump(qs, f, indent=2, ensure_ascii=False)
                        except Exception:
                            pass
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
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
