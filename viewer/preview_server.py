import http.server
import socketserver
import webbrowser
import json
import os
from pathlib import Path

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

                # Load existing
                verifications = {}
                if VERIFICATIONS_FILE.exists():
                    try:
                        with open(VERIFICATIONS_FILE, 'r') as f:
                            verifications = json.load(f)
                    except: pass

                key = f"{paper_id}_{q_num}"
                if status == 'pending':
                    if key in verifications:
                        del verifications[key]
                else:
                    verifications[key] = {
                        "status": status,
                        "diagramPath": diag_path
                    }

                # Save
                VERIFICATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(VERIFICATIONS_FILE, 'w') as f:
                    json.dump(verifications, f, indent=2)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
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
