"""
Root entry point to run the NTA Question Scraper & Dataset Extractor CLI.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "extractors"))

from nta_extractor.cli import main

if __name__ == "__main__":
    main()
