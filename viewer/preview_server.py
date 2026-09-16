"""
Local web server to browse extracted questions and preview diagrams and LaTeX formatting.
Run: python preview_server.py
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
REPO_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.name == "viewer") else Path(__file__).resolve().parent
DIRECTORY = REPO_ROOT


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


def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/viewer/question_viewer.html"
        print(f"\n========================================================")
        print(f"  Local Question & Diagram Viewer running at:")
        print(f"  --> {url}")
        print(f"  Press Ctrl+C to stop the server.")
        print(f"========================================================\n")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == "__main__":
    start_server()
