"""
Utility script to process flagged questions from data/verifications.json.
Runs specifically to review, export, or handle questions marked 'needs_fix'.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
VERIFICATIONS_FILE = DATA_DIR / "verifications.json"


def load_verifications():
    if not VERIFICATIONS_FILE.exists():
        print(f"No verifications file found at {VERIFICATIONS_FILE}")
        return {}
    try:
        with open(VERIFICATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading verifications: {e}")
        return {}


def process_flagged_questions():
    verifications = load_verifications()
    if not verifications:
        print("No verification records found.")
        return

    flagged = {k: v for k, v in verifications.items() if v.get("status") == "needs_fix"}
    verified = {k: v for k, v in verifications.items() if v.get("status") == "verified"}

    print("==================================================")
    print(f" Human Evaluation Report")
    print("==================================================")
    print(f" Total Verified   : {len(verified)}")
    print(f" Total Flagged    : {len(flagged)} (Needs Fix)")
    print("--------------------------------------------------")

    if flagged:
        print("\nFlagged Questions List:")
        for key in flagged:
            print(f" - {key}")
    print("==================================================")


if __name__ == "__main__":
    process_flagged_questions()
