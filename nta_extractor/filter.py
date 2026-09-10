"""
Paper filtering and exam categorization logic.
"""

import json
from pathlib import Path
from typing import List, Optional
from nta_extractor.config import TargetExam
from nta_extractor.models import PaperMetadata


def classify_exam(title: str) -> Optional[TargetExam]:
    """
    Classify paper into TargetExam category based on title heuristics.
    Category rules:
    - NEET_UG: Contains 'NEET'
    - MHT_CET: Contains 'KCET' or 'CET' (and not NEET/JEE specific)
    - JEE_MAIN: Contains 'BTECH', 'PAPER 1', 'PAPER1', or 'JEE'
    """
    upper_title = title.upper()

    # Rule 1: NEET UG
    if "NEET" in upper_title:
        return TargetExam.NEET_UG

    # Rule 2: MHT_CET / KCET
    if "KCET" in upper_title or "MHT-CET" in upper_title or "MHT_CET" in upper_title or "CET" in upper_title:
        # Avoid false positives with other exams like CUET/CUCET if needed,
        # but match CET / KCET as specified
        return TargetExam.MHT_CET

    # Rule 3: JEE MAIN
    if (
        "BTECH" in upper_title
        or "PAPER 1" in upper_title
        or "PAPER1" in upper_title
        or "JEE" in upper_title
    ):
        return TargetExam.JEE_MAIN

    return None


def load_and_filter_papers(
    json_path: str,
    target_exam: TargetExam = TargetExam.ALL,
    skip_category_labels: bool = True,
) -> List[PaperMetadata]:
    """
    Loads papers from JSON and filters them by target exam.

    Args:
        json_path: Path to nta_papers.json
        target_exam: TargetExam filter (ALL, JEE_MAIN, NEET_UG, MHT_CET)
        skip_category_labels: Whether to skip pure exam-category placeholders (e.g. title: 'NEET', 'JEE-Main')
                              if they duplicate paper IDs or lack paper shift details.

    Returns:
        List of filtered PaperMetadata objects.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[PaperMetadata] = []
    seen_keys = set()

    for item in data:
        paper_id = str(item.get("paperId", "")).strip()
        title = str(item.get("title", "")).strip()

        if not paper_id or not title:
            continue

        exam_cat = classify_exam(title)
        if not exam_cat:
            continue

        if skip_category_labels:
            pure_categories = {"JEE-MAIN", "NEET", "KCET", "CUCET", "NCET", "UGC-NET", "CUET(UG)", "CUET (PG)"}
            if title.strip().upper() in pure_categories:
                continue

        if target_exam != TargetExam.ALL and exam_cat != target_exam:
            continue

        dedup_key = (paper_id, title)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        results.append(
            PaperMetadata(
                paperId=paper_id,
                title=title,
                targetExam=exam_cat.value,
            )
        )

    return results
