"""
Subject classification and question metadata formatting.
"""

from typing import List, Optional
from nta_extractor.config import (
    DEFAULT_MARKS,
    DEFAULT_NEGATIVE_MARKS,
    DEFAULT_QUESTION_TYPE,
    TargetExam,
)
from nta_extractor.models import ExtractedQuestion, QuestionOption


def classify_subject(question_number: int, target_exam: str) -> str:
    """
    Categorizes question into subject based on question number and target exam:
    - Questions 1 - 30: Physics
    - Questions 31 - 60: Chemistry
    - Questions 61 - 90: Mathematics (JEE/CET) or Biology (NEET)
    - Questions > 90: Mathematics (JEE/CET) or Biology (NEET)
    """
    if question_number <= 30:
        return "Physics"
    elif question_number <= 60:
        return "Chemistry"
    else:
        if target_exam == TargetExam.NEET_UG.value:
            return "Biology"
        return "Mathematics"


def build_question_record(
    paper_id: str,
    paper_title: str,
    target_exam: str,
    question_number: int,
    image_url: str,
    local_image_path: str,
    correct_option: Optional[str] = None,
    marks: float = DEFAULT_MARKS,
    negative_marks: float = DEFAULT_NEGATIVE_MARKS,
    question_type: str = DEFAULT_QUESTION_TYPE,
) -> ExtractedQuestion:
    """
    Constructs a standardized ExtractedQuestion object strictly conforming to schema.
    """
    subject = classify_subject(question_number, target_exam)
    topic = f"{paper_title} - Q{question_number}"

    correct_options = [str(correct_option)] if correct_option else ["1"]

    options = [
        QuestionOption(key="1", text="Option 1"),
        QuestionOption(key="2", text="Option 2"),
        QuestionOption(key="3", text="Option 3"),
        QuestionOption(key="4", text="Option 4"),
    ]

    return ExtractedQuestion(
        paperId=paper_id,
        paperTitle=paper_title,
        targetExam=target_exam,
        subject=subject,
        questionNumber=question_number,
        topic=topic,
        imageUrl=image_url,
        localImagePath=local_image_path,
        type=question_type,
        options=options,
        correctOptions=correct_options,
        marks=marks,
        negativeMarks=negative_marks,
    )
