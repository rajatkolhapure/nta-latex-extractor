"""
High-performance session-based scraper engine using requests and BeautifulSoup.
"""

import re
from typing import List, Optional
import bs4
import requests

from nta_extractor.config import (
    BASE_URL,
    DEFAULT_HEADERS,
    INSTRUCTIONS_URL,
    LOGIN_URL,
    PAPER_URL,
    QUIZ_URL,
    REQUEST_TIMEOUT,
    SELECT_PAPER_URL,
)
from nta_extractor.engines.base import BaseScraperEngine, ScrapedQuestionCandidate


class SessionScraperEngine(BaseScraperEngine):
    """
    Direct HTTP session scraper simulating browser navigation handshakes.
    Fast, lightweight, and executes in ~1 second per paper without browser overhead.
    """

    def __init__(self, timeout: int = REQUEST_TIMEOUT):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._initialized = False

    def _create_session(self) -> requests.Session:
        """Creates a freshly initialized session with NTA portal."""
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        try:
            s.get(QUIZ_URL, timeout=self.timeout)
            s.get(
                f"{SELECT_PAPER_URL}?ID=1",
                headers={"X-Requested-With": "XMLHttpRequest"},
                timeout=self.timeout,
            )
        except Exception:
            pass
        return s

    def extract_paper_questions(self, paper_id: str) -> List[ScrapedQuestionCandidate]:
        """
        Navigates through Login -> Instructions -> Paper view and extracts question images.
        """
        session = self._create_session()

        # Step 1: Login with paper ID
        login_url = f"{LOGIN_URL}?PaperID={paper_id}"
        session.get(
            login_url,
            headers={"Referer": QUIZ_URL},
            timeout=self.timeout,
        )

        # Step 2: Instructions page
        session.get(
            INSTRUCTIONS_URL,
            headers={"Referer": login_url},
            timeout=self.timeout,
        )

        # Step 3: Access paper view (Language 1 = English by default)
        paper_url = f"{PAPER_URL}?Lng=1"
        resp = session.get(
            paper_url,
            headers={"Referer": INSTRUCTIONS_URL},
            timeout=self.timeout,
        )

        if resp.status_code != 200 or not resp.text:
            return []

        # Parse DOM with BeautifulSoup
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        img_tags = soup.find_all("img", src=lambda s: bool(s and "/Uploads/Question/" in s))

        candidates: List[ScrapedQuestionCandidate] = []
        seen_numbers = set()

        for idx, img in enumerate(img_tags):
            src = img.get("src", "").strip()
            if not src:
                continue

            # Ensure absolute URL
            if src.startswith("/"):
                image_url = f"{BASE_URL}{src}"
            elif not src.startswith("http"):
                image_url = f"{BASE_URL}/{src}"
            else:
                image_url = src

            # Extract question number from data attribute or parent text
            raw_qno = img.get("data-questionno")
            if raw_qno and raw_qno.isdigit():
                q_num = int(raw_qno)
            else:
                # Try to find Question X: in headings or fallback to sequential index
                parent_panel = img.find_parent("div", class_=lambda c: c and "panel" in c)
                title_elem = parent_panel.find(class_="question-title") if parent_panel else None
                if title_elem:
                    match = re.search(r"Question\s+(\d+)", title_elem.get_text(), re.I)
                    q_num = int(match.group(1)) if match else (idx + 1)
                else:
                    q_num = idx + 1

            if q_num in seen_numbers:
                continue
            seen_numbers.add(q_num)

            # Extract correct answer & marks from parent container if present
            parent_container = img.find_parent("table") or img.find_parent("div")
            correct_ans = None
            marks = None
            negative_marks = None

            if parent_container:
                ans_input = parent_container.find("input", class_="hdfCurrectAns")
                if ans_input and ans_input.get("value"):
                    val = ans_input.get("value").strip()
                    if val:
                        correct_ans = val

                marks_input = parent_container.find("input", class_="hdfCorrectAnsMarks")
                if marks_input and marks_input.get("value"):
                    try:
                        marks = float(marks_input.get("value"))
                    except (ValueError, TypeError):
                        pass

                neg_input = parent_container.find("input", class_="hdfInCorrectAnsMarks")
                if neg_input and neg_input.get("value"):
                    try:
                        # Value may be negative like -1, convert to positive magnitude for schema
                        val = abs(float(neg_input.get("value")))
                        negative_marks = val
                    except (ValueError, TypeError):
                        pass

            candidates.append(
                ScrapedQuestionCandidate(
                    questionNumber=q_num,
                    imageUrl=image_url,
                    correctOption=correct_ans,
                    marks=marks,
                    negativeMarks=negative_marks,
                )
            )

        # Sort strictly by question number
        candidates.sort(key=lambda c: c.questionNumber)
        session.close()
        return candidates

    def close(self) -> None:
        """Closes the underlying requests session."""
        self.session.close()
