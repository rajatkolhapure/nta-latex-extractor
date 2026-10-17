---
name: cbt-development
description: Computer-Based Test (CBT) architectures, NTA portal interaction, question schema, and exam conventions discovered in this project.
---

# Computer-Based Test (CBT) & Exam Architecture Guide

## Scope & Present Architecture

This codebase specializes in the extraction, digitization, classification, and inspection of Computer-Based Test (CBT) question papers from two sources:
1. **NTA Mock Test Portal Scraper** (`extractors/nta_extractor/`): Automated retrieval of official JEE Main and NEET UG papers from the online CBT system.
2. **Paper PDF Extractor** (`extractors/pdf_latex_extractor.py`): High-DPI digitization of MHT-CET and JEE question papers.
3. **Question & LaTeX Explorer** (`viewer/question_viewer.html`): Interactive web explorer for reviewing digitized CBT items.

> **Important Boundary**: The repository currently implements the ingestion, schema normalization, and inspection layers. It does NOT contain a student-facing live exam-taking runner (such as an active countdown timer, answer submission state machine, or result calculator). Do not assume or invent such modules unless explicitly requested.

---

## NTA CBT Portal Architecture (`extractors/nta_extractor/`)

The scraper package interacts directly with the NTA mock test web system (`https://nta.ac.in/Quiz`):

### 1. Portal Navigation Endpoints & Flow
- `GET /Quiz`: Landing page establishing initial ASP.NET session cookies.
- `GET /Quiz/Home/SelectPaper?ID=1`: AJAX endpoint returning paper dropdown options (`X-Requested-With: XMLHttpRequest`).
- `GET /Quiz/Home/Login?PaperID=<paper_id>`: Simulates test candidate login for a specific paper.
- `GET /Quiz/Home/Instructions`: Displays official CBT instructions and exam guidelines.
- `GET /Quiz/Home/Paper?Lng=1`: Question paper interface displaying all questions in English (`Lng=1`).

### 2. Dual Scraping Engines
- **`SessionScraperEngine` (`engines/session_engine.py`)**:
  - Uses `requests.Session` and `BeautifulSoup4`.
  - Simulates browser navigation headers (`User-Agent`, `Referer` chain) without rendering overhead.
  - Completes paper ingestion in ~1 second per paper.
- **`BrowserScraperEngine` (`engines/browser_engine.py`)**:
  - Uses Playwright Chromium in headless mode.
  - Automates real DOM navigation for testing or when JavaScript-rendered elements are required.

### 3. CBT Hidden Form Fields
The NTA portal DOM embeds hidden metadata inputs within question containers:
- `hdfCurrectAns`: The official answer key value (e.g., `"1"`, `"2"`, `"3"`, `"4"`).
- `hdfCorrectAnsMarks`: Points awarded for a correct response (default: `4.0` for JEE, `2.0` for CET Mathematics, `1.0` for CET Physics/Chemistry).
- `hdfInCorrectAnsMarks`: Negative marks deducted for an incorrect response (default: `1.0` for JEE, `0.0` for MHT-CET).

---

## Subject & Section Classification Rules

Standard Indian national and state engineering/medical entrance exams follow fixed question-numbering ranges:

### JEE Main / MHT-CET Numbering Convention
Implemented in `extractors/nta_extractor/classifier.py`:
- **Questions 1 – 30**: Physics
- **Questions 31 – 60**: Chemistry
- **Questions 61 – 90+**: Mathematics (JEE Main / MHT-CET) or Biology (NEET UG)

### Marking Schemes by Exam
- **JEE Main**:
  - Positive Marks: `4.0`
  - Negative Marks: `1.0`
- **MHT-CET**:
  - Mathematics: Positive `2.0`, Negative `0.0`
  - Physics & Chemistry: Positive `1.0`, Negative `0.0`
- **NEET UG**:
  - Positive Marks: `4.0`
  - Negative Marks: `1.0`

---

## Standardized CBT Question Schema

Every extracted question record across all extractors adheres to a standardized 26-field dictionary:

| Field Name | Type | Description |
|---|---|---|
| `paperId` | string | Unique paper identifier (e.g. `"133"`, or PDF stem name) |
| `paperTitle` | string | Human-readable exam paper title |
| `targetExam` | string | Target examination (`"JEE_MAIN"`, `"NEET_UG"`, `"MHT_CET"`) |
| `subject` | string | Exactly `"Physics"`, `"Chemistry"`, or `"Mathematics"` |
| `questionNumber` | integer | Question index within the paper |
| `topic` | string | Chapter or syllabus unit (e.g. `"Electrostatics"`, `"Trigonometric Functions"`) |
| `imageUrl` | string | Original remote or rendered page image URL |
| `localImagePath` | string | Path to downloaded question image or rendered PDF page |
| `type` | string | Standard question type (currently `"SINGLE_CHOICE"`) |
| `options` | array | 4 standardized option objects: `[{"key": "1", "text": "Option 1"}, ...]` |
| `correctOptions` | array | Array of correct option string keys (e.g. `["2"]`) |
| `marks` | float | Positive mark weight (e.g. `4.0`, `2.0`, `1.0`) |
| `negativeMarks` | float | Negative penalty (e.g. `1.0`, `0.0`) |
| `subTopic` | string | Specific tested sub-concept (e.g. `"Dimensional Analysis"`) |
| `difficulty` | string | Exactly `"Easy"`, `"Medium"`, or `"Hard"` |
| `exam` | string | Exam classification identifier |
| `latexQuestion` | string | Digitized question statement with LaTeX mathematical formatting |
| `hasTable` | boolean | Indicates whether the question body contains a data/observation table |
| `tableHtml` | string / null | Semantic HTML `<table>` transcription of data tables, or `null` |
| `latexOptions` | array | Digitized option objects: `[{"key": "1", "latex": "...", "hasDiagram": false}, ...]` |
| `hasDiagram` | boolean | Indicates whether the question body contains an extracted visual diagram |
| `diagramPath` | string / null | Local path to cropped diagram PNG, or `null` |
| `diagramBBox` | array / null | Pixel coordinates of the cropped diagram `[x1, y1, x2, y2]`, or `null` |
| `optionsHaveDiagrams` | boolean | True if any individual option contains a drawn structure |
| `optionDiagrams` | object | Dictionary mapping option keys to cropped diagram paths (`{"1": "...", ...}`) |
| `extractedVia` | string | Engine or model identifier (e.g. `"gemini-3.5-flash-lite"`) |

---

## State & Resumption Conventions

All batch operations must maintain strict idempotency and checkpointing:
- Before processing a question or page, check if its unique key (`f"{paperId}_q{qNum}"` or `f"{paperId}_p{page}_q{qNum}"`) already exists in `data/extracted_<subject>.json`.
- Progress must be saved immediately to the target subject file after each successful extraction to survive interruptions or API rate limits.
- On resumption, existing records must be loaded and preserved without alteration.
