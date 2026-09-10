"""
Command Line Interface for NTA Question Scraper & Dataset Extractor.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from nta_extractor.classifier import build_question_record
from nta_extractor.config import (
    DEFAULT_DOWNLOAD_CONCURRENCY,
    DEFAULT_IMAGE_DIR,
    DEFAULT_INPUT_FILE,
    DEFAULT_MARKS,
    DEFAULT_NEGATIVE_MARKS,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_QUESTION_TYPE,
    TargetExam,
)
from nta_extractor.downloader import download_single_image
from nta_extractor.engines.base import BaseScraperEngine
from nta_extractor.engines.browser_engine import BrowserScraperEngine
from nta_extractor.engines.session_engine import SessionScraperEngine
from nta_extractor.filter import load_and_filter_papers
from nta_extractor.models import ExtractedQuestion, PaperMetadata

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


def save_dataset_atomically(questions: List[ExtractedQuestion], output_path: str) -> None:
    """
    Saves the dataset atomically to avoid corrupting output file on unexpected exit.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = [q.model_dump() for q in questions]

    # Write to temp file in same directory then replace
    temp_file = target.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if temp_file.exists():
        temp_file.replace(target)


def run_extractor(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    image_dir: str = DEFAULT_IMAGE_DIR,
    exam_filter: str = "ALL",
    paper_id_filter: Optional[str] = None,
    limit: Optional[int] = None,
    engine_name: str = "session",
    headless: bool = True,
    concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    dry_run: bool = False,
    marks: float = DEFAULT_MARKS,
    negative_marks: float = DEFAULT_NEGATIVE_MARKS,
) -> None:
    """Main extractor execution workflow."""
    console.print(
        Panel.fit(
            "[bold cyan]NTA Question Scraper & Dataset Extractor CLI[/bold cyan]\n"
            "[dim]Automated mock paper scraper and dataset compiler for NTA portal[/dim]",
            border_style="cyan",
        )
    )

    target_exam = TargetExam(exam_filter.upper())
    papers = load_and_filter_papers(input_file, target_exam=target_exam)

    if paper_id_filter:
        papers = [p for p in papers if p.paperId == str(paper_id_filter).strip()]

    if limit and limit > 0:
        papers = papers[:limit]

    if not papers:
        console.print(
            f"[bold yellow]No papers found matching criteria (Exam: {exam_filter}, Paper ID: {paper_id_filter}).[/bold yellow]"
        )
        return

    # Display papers summary table
    table = Table(title=f"Target Papers ({len(papers)} Total)", header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Paper ID", style="cyan", width=10)
    table.add_column("Target Exam", style="green", width=14)
    table.add_column("Paper Title", style="white")

    for idx, p in enumerate(papers, 1):
        table.add_row(str(idx), p.paperId, p.targetExam or "UNKNOWN", p.title)

    console.print(table)

    if dry_run:
        console.print("\n[bold yellow]Dry run mode active. No downloads or scraping performed.[/bold yellow]")
        return

    # Initialize Engine
    console.print(f"\n[bold]Initializing scraping engine:[/bold] [green]{engine_name.upper()}[/green]")
    engine: BaseScraperEngine
    if engine_name.lower() == "browser":
        engine = BrowserScraperEngine(headless=headless)
    else:
        engine = SessionScraperEngine()

    all_questions: List[ExtractedQuestion] = []
    # If output file already exists, load existing questions to allow resume/extend
    if Path(output_file).exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_raw = json.load(f)
                all_questions = [ExtractedQuestion(**item) for item in existing_raw]
                console.print(f"[dim]Loaded {len(all_questions)} existing questions from {output_file}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse existing output file ({e}), starting fresh.[/yellow]")
            all_questions = []

    # Map existing (paperId, questionNumber) to avoid duplicates
    existing_keys = {(q.paperId, q.questionNumber) for q in all_questions}

    total_downloaded = 0
    total_skipped = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            paper_task = progress.add_task("[cyan]Processing papers...", total=len(papers))

            for paper in papers:
                progress.update(paper_task, description=f"[cyan]Paper {paper.paperId}: {paper.title[:30]}...")

                try:
                    candidates = engine.extract_paper_questions(paper.paperId)
                except Exception as e:
                    console.print(f"[red]Error scraping paper {paper.paperId} ({paper.title}): {e}[/red]")
                    progress.advance(paper_task)
                    continue

                if not candidates:
                    console.print(f"[yellow]No question images found for Paper ID {paper.paperId}[/yellow]")
                    progress.advance(paper_task)
                    continue

                paper_questions: List[ExtractedQuestion] = []

                for cand in candidates:
                    # Download image
                    local_img_path, success = download_single_image(
                        image_url=cand.imageUrl,
                        output_dir=image_dir,
                    )
                    if success:
                        total_downloaded += 1
                    else:
                        total_skipped += 1

                    # Use extracted marks or CLI defaults
                    q_marks = cand.marks if cand.marks is not None else marks
                    q_neg_marks = cand.negativeMarks if cand.negativeMarks is not None else negative_marks

                    question_record = build_question_record(
                        paper_id=paper.paperId,
                        paper_title=paper.title,
                        target_exam=paper.targetExam or "JEE_MAIN",
                        question_number=cand.questionNumber,
                        image_url=cand.imageUrl,
                        local_image_path=local_img_path,
                        correct_option=cand.correctOption,
                        marks=q_marks,
                        negative_marks=q_neg_marks,
                    )

                    key = (paper.paperId, cand.questionNumber)
                    if key in existing_keys:
                        # Update record if existing
                        all_questions = [q for q in all_questions if (q.paperId, q.questionNumber) != key]
                    existing_keys.add(key)
                    paper_questions.append(question_record)

                all_questions.extend(paper_questions)
                # Sort master list by targetExam, paperId, questionNumber
                all_questions.sort(key=lambda q: (q.targetExam, q.paperId, q.questionNumber))

                # Atomic intermediate save
                save_dataset_atomically(all_questions, output_file)

                console.print(
                    f"  [green][+][/green] Extracted [bold]{len(candidates)}[/bold] questions from [bold]{paper.title}[/bold] (Paper ID: {paper.paperId})"
                )
                progress.advance(paper_task)

    finally:
        engine.close()

    # Final Summary
    save_dataset_atomically(all_questions, output_file)

    summary_table = Table(title="Extraction Complete", header_style="bold green")
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="bold white")

    summary_table.add_row("Total Papers Processed", str(len(papers)))
    summary_table.add_row("Total Questions in Dataset", str(len(all_questions)))
    summary_table.add_row("Images Download Folder", str(Path(image_dir).resolve()))
    summary_table.add_row("Dataset File", str(Path(output_file).resolve()))

    # Subject breakdown
    sub_counts = {}
    for q in all_questions:
        sub_counts[q.subject] = sub_counts.get(q.subject, 0) + 1

    for sub, count in sorted(sub_counts.items()):
        summary_table.add_row(f"Subject: {sub}", str(count))

    console.print("\n", summary_table)


def main() -> None:
    """CLI entrypoint with argument parser."""
    parser = argparse.ArgumentParser(
        description="Standalone NTA Question Scraper & Dataset Extractor CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_FILE,
        help="Path to input nta_papers.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output master JSON dataset",
    )
    parser.add_argument(
        "--image-dir",
        "-d",
        default=DEFAULT_IMAGE_DIR,
        help="Local directory to store downloaded question images",
    )
    parser.add_argument(
        "--exam",
        "-e",
        choices=["ALL", "JEE_MAIN", "NEET_UG", "MHT_CET"],
        default="ALL",
        help="Filter papers by target exam",
    )
    parser.add_argument(
        "--paper-id",
        "-p",
        default=None,
        help="Scrape a single specific paper by ID (e.g. 326)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of papers to scrape (ideal for testing)",
    )
    parser.add_argument(
        "--engine",
        choices=["session", "browser"],
        default="session",
        help="Scraping engine: 'session' (fast HTTP requests) or 'browser' (Playwright Chromium)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser engine in visible (non-headless) mode",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_DOWNLOAD_CONCURRENCY,
        help="Number of concurrent image download threads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect filtered papers without performing downloads or writes",
    )
    parser.add_argument(
        "--marks",
        type=float,
        default=DEFAULT_MARKS,
        help="Default marks per question",
    )
    parser.add_argument(
        "--negative-marks",
        type=float,
        default=DEFAULT_NEGATIVE_MARKS,
        help="Default negative marks per question",
    )

    args = parser.parse_args()

    run_extractor(
        input_file=args.input,
        output_file=args.output,
        image_dir=args.image_dir,
        exam_filter=args.exam,
        paper_id_filter=args.paper_id,
        limit=args.limit,
        engine_name=args.engine,
        headless=not args.no_headless,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        marks=args.marks,
        negative_marks=args.negative_marks,
    )


if __name__ == "__main__":
    main()
