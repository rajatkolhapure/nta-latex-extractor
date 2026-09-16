"""
Configuration and constants for the NTA Question Extractor CLI Tool.
"""

import os
from enum import Enum
from pathlib import Path


class TargetExam(str, Enum):
    JEE_MAIN = "JEE_MAIN"
    NEET_UG = "NEET_UG"
    MHT_CET = "MHT_CET"
    ALL = "ALL"


# Network Endpoints
BASE_URL = "https://nta.ac.in"
QUIZ_URL = f"{BASE_URL}/Quiz"
SELECT_PAPER_URL = f"{BASE_URL}/Quiz/Home/SelectPaper"
LOGIN_URL = f"{BASE_URL}/Quiz/Home/Login"
INSTRUCTIONS_URL = f"{BASE_URL}/Quiz/Home/Instructions"
PAPER_URL = f"{BASE_URL}/Quiz/Home/Paper"

# HTTP Headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": QUIZ_URL,
}

# Question Image Match Pattern
QUESTION_IMAGE_PATTERN = r"/Uploads/Question/"

# Default Local File Paths
DEFAULT_INPUT_FILE = "data/nta_papers.json"
DEFAULT_OUTPUT_FILE = "data/all_nta_questions.json"
DEFAULT_IMAGE_DIR = "./downloads/question_images"

# Question Defaults
DEFAULT_MARKS = 4.0
DEFAULT_NEGATIVE_MARKS = 1.0
DEFAULT_QUESTION_TYPE = "SINGLE_CHOICE"

# Retry & Timeout Configuration
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5
DEFAULT_DOWNLOAD_CONCURRENCY = 5
