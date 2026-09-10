"""
Playwright-based Chromium browser scraper engine.
"""

import re
from typing import List
from playwright.sync_api import sync_playwright

from nta_extractor.config import (
    BASE_URL,
    INSTRUCTIONS_URL,
    LOGIN_URL,
    PAPER_URL,
    QUIZ_URL,
    REQUEST_TIMEOUT,
)
from nta_extractor.engines.base import BaseScraperEngine, ScrapedQuestionCandidate


class BrowserScraperEngine(BaseScraperEngine):
    """
    Automated Chromium browser scraper using Playwright.
    Ideal for real browser fidelity and visual debugging.
    """

    def __init__(self, headless: bool = True, timeout: int = REQUEST_TIMEOUT):
        self.headless = headless
        self.timeout = timeout * 1000  # ms
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._initialized = False

    def _init_browser_session(self) -> None:
        """Initializes ASP.NET session state with NTA portal."""
        if self._initialized:
            return
        try:
            self._page.goto(QUIZ_URL, wait_until="domcontentloaded", timeout=self.timeout)
            self._context.request.get(
                f"https://nta.ac.in/Quiz/Home/SelectPaper?ID=1",
                headers={"X-Requested-With": "XMLHttpRequest", "Referer": QUIZ_URL},
            )
            self._initialized = True
        except Exception:
            pass

    def extract_paper_questions(self, paper_id: str) -> List[ScrapedQuestionCandidate]:
        """
        Navigates browser to Paper view and extracts question images from rendered DOM.
        """
        self._init_browser_session()
        page = self._page

        # Step 1: Login
        login_url = f"{LOGIN_URL}?PaperID={paper_id}"
        page.goto(login_url, wait_until="domcontentloaded", timeout=self.timeout, referer=QUIZ_URL)

        # Step 2: Instructions
        page.goto(INSTRUCTIONS_URL, wait_until="domcontentloaded", timeout=self.timeout, referer=login_url)

        # Step 3: Paper view
        paper_url = f"{PAPER_URL}?Lng=1"
        page.goto(paper_url, wait_until="domcontentloaded", timeout=self.timeout, referer=INSTRUCTIONS_URL)

        # Wait for question images to appear in DOM
        try:
            page.wait_for_selector("img[src*='/Uploads/Question/']", timeout=self.timeout)
        except Exception:
            # If no selector matches within timeout, return empty
            return []

        # Extract question details using DOM evaluation
        raw_items = page.evaluate(
            """() => {
                const imgs = Array.from(document.querySelectorAll("img[src*='/Uploads/Question/']"));
                return imgs.map((img, idx) => {
                    const qnoAttr = img.getAttribute("data-questionno");
                    let qno = qnoAttr ? parseInt(qnoAttr, 10) : (idx + 1);
                    
                    const parentContainer = img.closest("table") || img.closest(".panel") || img.parentElement;
                    let correctAns = null;
                    let marks = null;
                    let negMarks = null;
                    
                    if (parentContainer) {
                        const ansInput = parentContainer.querySelector(".hdfCurrectAns");
                        if (ansInput && ansInput.value) correctAns = ansInput.value.trim();
                        
                        const marksInput = parentContainer.querySelector(".hdfCorrectAnsMarks");
                        if (marksInput && marksInput.value) marks = parseFloat(marksInput.value);
                        
                        const negInput = parentContainer.querySelector(".hdfInCorrectAnsMarks");
                        if (negInput && negInput.value) negMarks = Math.abs(parseFloat(negInput.value));
                    }
                    
                    return {
                        questionNumber: qno,
                        imageUrl: img.src,
                        correctOption: correctAns,
                        marks: marks,
                        negativeMarks: negMarks
                    };
                });
            }"""
        )

        candidates: List[ScrapedQuestionCandidate] = []
        seen = set()

        for item in raw_items:
            q_num = item.get("questionNumber", 1)
            if q_num in seen:
                continue
            seen.add(q_num)

            candidates.append(
                ScrapedQuestionCandidate(
                    questionNumber=q_num,
                    imageUrl=item.get("imageUrl", ""),
                    correctOption=item.get("correctOption"),
                    marks=item.get("marks"),
                    negativeMarks=item.get("negativeMarks"),
                )
            )

        candidates.sort(key=lambda c: c.questionNumber)
        return candidates

    def close(self) -> None:
        """Shuts down browser and Playwright runtime."""
        try:
            self._page.close()
            self._context.close()
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass
