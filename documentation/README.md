# AutoViz AI - Premium Enterprise Visualization Dashboard

AutoViz AI is a premium, modern data visualization dashboard built on top of Streamlit. It allows you to transform raw datasets (CSV or Excel) into high-quality interactive or static charts using plain English prompts, automatic type-aware recommendations, and customizable color themes. It also features a built-in statistical insight profile and optional LLM-powered business reports.

## Features

1. **Dual Visualization Engines**:
   - **Interactive Canvas (Plotly)**: Enable zoomable, hover-supported, custom tool-tipped graphs.
   - **Static Canvas (Matplotlib)**: Deliver high-quality, customized publication-ready layouts.
2. **AI-Driven Parameter Configurator**:
   - Matches column attributes dynamically to X and Y dimensions using fuzzy similarity rules.
   - Identifies prompt intents (trends, distributions, compositions, correlations) and matches them to suitable chart templates.
   - Optional LLM-guided schema parsing (supporting Google Gemini & OpenAI API models).
3. **Statistical Profile Engine**:
   - Automatically computes aggregate metrics (mean, median, range, spread).
   - Identifies sequence trajectories (upward/downward timelines).
   - Runs IQR boundary checks to pinpoint anomalous values (outliers).
   - Optional LLM executive business analysis generator.
4. **Bespoke Theme Control**:
   - Support for custom Light and Dark modes.
   - Distinct color palettes (Cool Indigo, Emerald Garden, Sunset Orange, Dark Charcoal) matching the visual styling across all chart vectors.
5. **Instant Playground Loading**:
   - Dropdown options to instantly parse test datasets (Sales Trends, Student Marks, Demographics).

---

## Workspace Architecture

- [app.py](file:///d:/Projects/AutoViz_AI/app.py): Core dashboard visual layout, styling overrides, interactive settings, and coordinate binding.
- [chart_engine.py](file:///d:/Projects/AutoViz_AI/chart_engine.py): Generates interactive Plotly models and static Matplotlib frames customized with theme-specific palettes.
- [ai_engine.py](file:///d:/Projects/AutoViz_AI/ai_engine.py): Intent-based prompt extraction, fuzzy token mapping, and optional LLM parsing.
- [insight_engine.py](file:///d:/Projects/AutoViz_AI/insight_engine.py): Runs descriptive statistics, IQR outlier tracking, Pearson correlation, and builds the analytical Markdown reports.
- [utils.py](file:///d:/Projects/AutoViz_AI/utils.py): Holds type checkers, outlier detectors, and wrappers for Gemini/OpenAI API calls.
- [datasets/](file:///d:/Projects/AutoViz_AI/datasets): Preloaded CSV spreadsheets for immediate testing.

---

## Setup & Running Guide

Ensure Python 3.10+ is installed on your local environment.

### 1. Set Up the Virtual Environment
Create the virtual environment folder:
```powershell
python -m venv .venv
```

### 2. Install Dependencies
Activate the environment and install requirements:
```powershell
# On Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# On macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
Start the local Streamlit development server:
```powershell
streamlit run app.py
```

Open `http://localhost:8501` inside your web browser.

---

## Prompt Guides

Try entering these query strings into the AI prompt field:
- `"show sales trend over time"` (pre-configures a Line chart mapping Month to Sales)
- `"distribution of age data"` (pre-configures a Histogram showing the spread)
- `"compare student marks vs study hours"` (pre-configures a Scatter plot mapping StudyHours to Marks)
- `"composition of sales"` (pre-configures a Pie chart)
- `"compare students by marks"` (pre-configures a Bar chart comparing names to grades)
