"""
Sync tool: merges verified/re-extracted questions from data/re_extracted_flagged.json
directly back into master datasets (extracted_physics.json, extracted_chemistry.json, extracted_mathematics.json).
"""

import json
from pathlib import Path

DATA_DIR = Path("data")
RE_FLAGGED = DATA_DIR / "re_extracted_flagged.json"

subj_files = {
    "Physics": DATA_DIR / "extracted_physics.json",
    "Chemistry": DATA_DIR / "extracted_chemistry.json",
    "Mathematics": DATA_DIR / "extracted_mathematics.json",
}

if not RE_FLAGGED.exists():
    print("No re_extracted_flagged.json found.")
    exit(0)

re_records = json.load(open(RE_FLAGGED, encoding='utf-8'))
print(f"Loaded {len(re_records)} records from {RE_FLAGGED}")

masters = {}
for subj, fpath in subj_files.items():
    if fpath.exists():
        masters[subj] = json.load(open(fpath, encoding='utf-8'))

updated_count = 0
for rec in re_records:
    pid = str(rec.get("paperId"))
    qnum = str(rec.get("questionNumber"))
    subj = rec.get("subject", "Physics")
    
    if subj in masters:
        found = False
        for idx, item in enumerate(masters[subj]):
            if str(item.get("paperId")) == pid and str(item.get("questionNumber")) == qnum:
                masters[subj][idx] = rec
                updated_count += 1
                found = True
                break
        if not found:
            # Check other subjects in case of re-classification
            for other_s in masters:
                if other_s != subj:
                    for idx, item in enumerate(masters[other_s]):
                        if str(item.get("paperId")) == pid and str(item.get("questionNumber")) == qnum:
                            masters[other_s][idx] = rec
                            updated_count += 1
                            break

for subj, fpath in subj_files.items():
    if subj in masters:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(masters[subj], f, indent=2, ensure_ascii=False)

print(f"Successfully synchronized {updated_count} questions into master datasets!")
