"""
Base abstract interface for scraping engines.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel


class ScrapedQuestionCandidate(BaseModel):
    """Raw scraped question extracted from DOM before processing."""
    questionNumber: int
    imageUrl: str
    correctOption: Optional[str] = None
    marks: Optional[float] = None
    negativeMarks: Optional[float] = None


class BaseScraperEngine(ABC):
    """Abstract base class for scraping engines."""

    @abstractmethod
    def extract_paper_questions(self, paper_id: str) -> List[ScrapedQuestionCandidate]:
        """
        Scrapes question image URLs and metadata for a given paperId.

        Args:
            paper_id: NTA paper ID

        Returns:
            List of ScrapedQuestionCandidate
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes any open network/browser connections."""
        pass
