"""
Pydantic data models for NTA Question Extractor.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata of an NTA paper entry from nta_papers.json."""
    paperId: str
    title: str
    targetExam: Optional[str] = None


class QuestionOption(BaseModel):
    """Standardized option model."""
    key: str
    text: str


class ExtractedQuestion(BaseModel):
    """Compiled question model conforming strictly to target schema."""
    paperId: str
    paperTitle: str
    targetExam: str
    subject: str
    questionNumber: int
    topic: str
    imageUrl: str
    localImagePath: str
    type: str = "SINGLE_CHOICE"
    options: List[QuestionOption] = Field(
        default_factory=lambda: [
            QuestionOption(key="1", text="Option 1"),
            QuestionOption(key="2", text="Option 2"),
            QuestionOption(key="3", text="Option 3"),
            QuestionOption(key="4", text="Option 4"),
        ]
    )
    correctOptions: List[str] = Field(default_factory=lambda: ["1"])
    marks: float = 4.0
    negativeMarks: float = 1.0
