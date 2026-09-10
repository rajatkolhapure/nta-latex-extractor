"""
Local web server to browse extracted questions and preview diagrams and LaTeX formatting.
Run: python preview_server.py
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent.resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)


def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/question_viewer.html"
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
