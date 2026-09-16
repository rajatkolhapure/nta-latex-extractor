"""
Sorts and splits NTA questions into subject-wise datasets.
Generates:
  1. questions_by_subject.json: Grouped dictionary {"Physics": [...], "Chemistry": [...], "Mathematics": [...]}
  2. physics_questions.json: Dedicated Physics dataset
  3. chemistry_questions.json: Dedicated Chemistry dataset
  4. mathematics_questions.json: Dedicated Mathematics dataset
  5. all_nta_questions_sorted.json: All 2,999 questions sorted by Subject -> Paper -> Question Number
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INPUT_FILE = DATA_DIR / "all_nta_questions.json"


def sort_and_split_questions():
    if not INPUT_FILE.exists():
        if (REPO_ROOT / "all_nta_questions.json").exists():
            input_file = REPO_ROOT / "all_nta_questions.json"
        else:
            print(f"Error: {INPUT_FILE} not found.")
            return
    else:
        input_file = INPUT_FILE

    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {input_file.name}")

    # Standardize subject names
    subject_map = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "mathematics": "Mathematics",
        "maths": "Mathematics",
        "math": "Mathematics",
    }

    by_subject = defaultdict(list)
    for q in questions:
        raw_subj = str(q.get("subject", "Unclassified")).strip()
        standard_subj = subject_map.get(raw_subj.lower(), raw_subj.capitalize())
        q["subject"] = standard_subj
        by_subject[standard_subj].append(q)

    # Sort each subject list by Paper ID (integer if possible), then Question Number
    def sort_key(q):
        try:
            pid = int(q.get("paperId", 0))
        except (ValueError, TypeError):
            pid = 0
        try:
            qnum = int(q.get("questionNumber", 0))
        except (ValueError, TypeError):
            qnum = 0
        return (pid, qnum)

    for subj in by_subject:
        by_subject[subj].sort(key=sort_key)

    # 1. Export grouped dictionary
    grouped_path = DATA_DIR / "questions_by_subject.json"
    with open(grouped_path, "w", encoding="utf-8") as f:
        json.dump(dict(by_subject), f, indent=2, ensure_ascii=False)
    print(f" Saved: {grouped_path.name}")

    # 2. Export individual subject files
    for subj, q_list in by_subject.items():
        filename = f"{subj.lower()}_questions.json"
        out_path = DATA_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(q_list, f, indent=2, ensure_ascii=False)
        print(f" Saved: {filename} ({len(q_list)} questions)")

    # 3. Export flat array sorted by subject
    subject_order = ["Physics", "Chemistry", "Mathematics"]
    all_sorted = []
    for s in subject_order:
        if s in by_subject:
            all_sorted.extend(by_subject[s])
    for s, q_list in by_subject.items():
        if s not in subject_order:
            all_sorted.extend(q_list)

    sorted_path = DATA_DIR / "all_nta_questions_sorted.json"
    with open(sorted_path, "w", encoding="utf-8") as f:
        json.dump(all_sorted, f, indent=2, ensure_ascii=False)
    print(f" Saved: {sorted_path.name} ({len(all_sorted)} questions sorted by subject)")


if __name__ == "__main__":
    sort_and_split_questions()
