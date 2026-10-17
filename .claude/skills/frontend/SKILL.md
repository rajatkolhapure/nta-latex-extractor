---
name: frontend
description: Frontend architecture, styling, and rendering conventions for the NTA Question Viewer.
---

# Frontend Architecture and Development Guide

## Architecture Overview

The frontend is a lightweight, zero-build single-page web application residing in `viewer/question_viewer.html` and served by `viewer/preview_server.py`. It is designed to run locally in any browser with instant reload, requiring no Node.js, npm, or build step.

## Technology Stack

- **Markup & Scripting**: Vanilla HTML5 and ES6 JavaScript (single-file architecture).
- **Styling**: Tailwind CSS loaded via CDN (`https://cdn.tailwindcss.com`).
- **Math Engine**: KaTeX loaded via CDN (`katex.min.css`, `katex.min.js`, and `contrib/auto-render.min.js`).
- **Web Server**: `viewer/preview_server.py` using Python standard library `http.server` and `socketserver` (default port: 8000).

## Key Components in `viewer/question_viewer.html`

### 1. Top Control Bar (`<header>`)
- **Theme Toggle Button (`#themeToggleBtn`)**: Switches between dark theme (default) and light theme (`.light-theme` class applied to `document.body`). Saves preference in `localStorage.getItem('viewer_theme')`.
- **Dataset Selector (`#datasetSelector`)**: Dropdown switching between JSON files in `data/` (`data/extracted_physics.json`, `data/extracted_chemistry.json`, `data/extracted_mathematics.json`, `data/test_q30.json`, `data/all_nta_questions.json`). Triggers `switchDataset()` which calls `loadDataset()`.
- **Subject Filter (`#subjectFilter`)**: Filters currently loaded questions by `ALL`, `Physics`, `Chemistry`, or `Mathematics`. Triggers `applyFilters()`.
- **Search Input (`#searchInput`)**: Real-time substring filter on `topic` and `paperId`.

### 2. Pagination & Metrics Bar
- **Counter (`#statsText`)**: Displays total matching questions and current filter state.
- **Navigation Controls (`#prevBtn`, `#nextBtn`, `#pageIndicator`)**: 10 questions per page (`pageSize = 10`). State managed via `currentPage` variable.

### 3. Question Feed Container (`#questionsContainer`)
- Questions rendered dynamically via template literals inside `renderQuestions()`.
- After DOM insertion, `renderMathInElement(container, options)` is invoked to parse and render KaTeX math spans.

## Question Card Anatomy & DOM Structure

Each card renders inside a `.card-bg` container with the following sections:
1. **Header Metadata**:
   - Exam badge (`JEE_MAIN` or `MHT_CET`)
   - Subject badge with subject-specific border color (e.g. blue for Physics, emerald for Chemistry, amber for Mathematics)
   - Question number badge (`Q{number}`)
   - Difficulty badge (`Easy` = emerald, `Medium` = amber, `Hard` = rose)
   - Topic and Subtopic badges
   - Paper identifier (`Paper {paperId}`)
   - Model badge (`extractedVia`) and marks display (`+{marks} / -{negativeMarks}`)
2. **Question Statement**:
   - For extracted questions: `.latex-content` with `whitespace-pre-line` and KaTeX rendering.
   - For raw scans: centered image tag with fallback to original scan.
3. **Observation Table (`.table-container`)**:
   - Conditional rendering when `q.hasTable` is true and `q.tableHtml` exists.
   - Renders semantic `<table>` wrapped inside `.table-wrapper` with horizontal scroll support.
4. **Question Diagram Frame (`.diagram-frame`)**:
   - Conditional rendering when `q.hasDiagram` is true and `q.diagramPath` exists.
   - Image rendered with `.diagram-img` and `loading="lazy"`.
5. **Options Grid**:
   - Responsive 2-column layout (`grid grid-cols-1 md:grid-cols-2 gap-3`).
   - Each option displays key badge (`1`, `2`, `3`, `4`).
   - If option contains a drawn diagram (`q.optionDiagrams[key]`), renders an option-level diagram frame.
   - If option matches `correctOptions`, highlights with emerald border and an "Answer Key" badge.

## Media and Asset Resolution Rules

All media paths must pass through the `resolveMediaSrc(path)` JavaScript helper before being assigned to `src`:
- Strips Windows backslashes (`\` -> `/`).
- Strips absolute `file:///` prefixes up to the `/downloads/` directory.
- Prepends `/` to relative paths so assets resolve from the server root regardless of current URL depth.
- Prevents broken image links when served under `http://localhost:8000/viewer/question_viewer.html`.

## Dataset Loading Rules

The `loadDataset()` function implements a multi-path candidate fallback:
- Evaluates `currentDataset`, `/${currentDataset}`, `/data/${basename}`, `../data/${basename}`, and `data/${basename}`.
- Allows the viewer to work seamlessly when served via `preview_server.py`, alternative local servers, or direct browser file inspection.

## KaTeX Configuration Rules

- Delimiters configured in `renderMathInElement`:
  - `$$...$$` (display math)
  - `$...$` (inline math)
  - `\(...\)` (inline math)
  - `\[...\]` (display math)
- `throwOnError` must be set to `false` to prevent rendering crashes on invalid or complex OCR output.

## Styling & Responsiveness Guidelines

- Do not introduce CSS preprocessors (SASS, LESS) or build tools.
- Dark theme uses slate palette (`bg-slate-900`, `bg-slate-800`, `border-slate-700`, `text-slate-200`).
- Light theme overrides are grouped under `body.light-theme` in the `<style>` block.
- Maintain responsive breakpoints using standard Tailwind classes (`md:`, `sm:`).
- Diagram images use `max-h-72 object-contain` to preserve aspect ratios and avoid layout shifting.
