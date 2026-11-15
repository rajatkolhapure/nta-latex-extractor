"""
Step 1 ONLY: Delete MHT-CET flagged PDF-extracted questions from all datasets
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MHT_DELETE_NUMS = [9, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
MHT_PAPER_ID = "MHT-CET Part Test -1Ans key"

DATASETS = [
    "extracted_physics.json",
    "extracted_chemistry.json",
    "extracted_mathematics.json",
    "re_extracted_flagged.json",
]

print("=" * 60)
print("Deleting MHT-CET PDF-extracted flagged questions")
print("=" * 60)

for dataset_file in DATASETS:
    path = DATA_DIR / dataset_file
    if not path.exists():
        print(f"  [skip] {dataset_file} not found")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    before = len(data)
    filtered = []
    for q in data:
        pid = str(q.get("paperId", ""))
        qnum = q.get("questionNumber")
        if pid == MHT_PAPER_ID and qnum in MHT_DELETE_NUMS:
            print(f"  x DELETE [{dataset_file}] Q{qnum}")
        else:
            filtered.append(q)
    after = len(filtered)
    if after < before:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"  -> {dataset_file}: {before} -> {after} (-{before - after})")
    else:
        print(f"  -> {dataset_file}: nothing to delete")

# Update verifications.json
verif_path = DATA_DIR / "verifications.json"
if verif_path.exists():
    with open(verif_path, "r", encoding="utf-8") as f:
        verifs = json.load(f)
    removed = []
    for n in MHT_DELETE_NUMS:
        k = f"{MHT_PAPER_ID}_{n}"
        if k in verifs:
            del verifs[k]
            removed.append(k)
    if removed:
        with open(verif_path, "w", encoding="utf-8") as f:
            json.dump(verifs, f, indent=2, ensure_ascii=False)
        print(f"  -> Removed {len(removed)} verification entries")
    else:
        print("  -> No verification entries to remove")

print()
print("DONE - Deletion complete.")
print(f"Now run step 2 with: python tools/step2_reextract.py --key YOUR_API_KEY")
