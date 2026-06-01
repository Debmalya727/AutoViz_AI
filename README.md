# AutoViz AI - Enterprise Data Visualization & ML Studio

AutoViz AI is a premium, modern data visualization and analytics dashboard built on Streamlit. It allows you to transform raw datasets (CSV or Excel) into high-quality interactive 3D/2D plots or static publication-ready charts using plain English prompts, automatic type-aware recommendations, and customizable color themes. It also features a built-in SQL console, unsupervised ML modeling (K-Means, PCA projections), and descriptive analytics.

# Live link : https://lightautoviz.streamlit.app/
---

## 🌟 Key Features

1. **Dual Visualization Engines**:
   - **Interactive Canvas (Plotly)**: Features zoomable, hover-supported 2D charts and rotatable **interactive 3D Scatter & Line plots**.
   - **Static Canvas (Matplotlib)**: Deliver high-quality, customized publication-ready layouts (including 3D subplots).
2. **AI-Driven Parameter Configurator**:
   - Matches column attributes dynamically to X, Y, and Z dimensions using fuzzy similarity rules.
   - Identifies prompt intents (trends, distributions, compositions, correlations) and matches them to suitable chart templates.
   - Optional LLM-guided schema parsing (supporting Google Gemini & OpenAI API models).
3. **Machine Learning Studio**:
   - **K-Means Clustering**: Group rows according to similarities and plot them on interactive 3D colored fields.
   - **PCA Dimensionality Reduction**: Standardize and project high-dimensional inputs into 3 Principal Components (PC1, PC2, PC3) to visualize variance structure.
   - **Linear Regression Fit**: Fit a prediction model, plot the regression line, and output formula constants.
4. **SQLite Transformation Console**:
   - Write standard SQL statements (e.g. `SELECT * FROM data WHERE Sales > 20000 ORDER BY Sales DESC`) against your active dataset. The results dynamically feed into the visualization canvas and ML studio.
5. **Theme & Palette Controls**:
   - Customizable Light/Dark modes.
   - Four premium color palettes (Cool Indigo, Emerald Garden, Sunset Orange, Dark Charcoal) matching the visual styling across all chart vectors.

---

## 📂 Workspace Architecture

- **[app.py](file:///d:/Projects/AutoViz_AI/app.py)**: Main entrypoint. Contains multi-tab dashboard layouts, custom neumorphic/glassmorphic CSS styles, and sync bindings.
- **[chart_engine.py](file:///d:/Projects/AutoViz_AI/chart_engine.py)**: Plotly and Matplotlib 3D/2D graph generator styled with custom color palettes.
- **[ml_engine.py](file:///d:/Projects/AutoViz_AI/ml_engine.py)**: Machine learning algorithms (K-Means clustering, PCA projections, Linear Regression line fitting).
- **[data_engine.py](file:///d:/Projects/AutoViz_AI/data_engine.py)**: Temporary SQLite bridge to execute SQL queries on the loaded data.
- **[ai_engine.py](file:///d:/Projects/AutoViz_AI/ai_engine.py)**: Intent-based prompt extraction, fuzzy token mapping, and optional LLM query parsing.
- **[insight_engine.py](file:///d:/Projects/AutoViz_AI/insight_engine.py)**: Computes descriptive analytics, Pearson correlation, and IQR boundary checks.
- **[utils.py](file:///d:/Projects/AutoViz_AI/utils.py)**: Shared utility helpers (outlier detection, Gemini/OpenAI API wrappers, data column categorizers).
- **[requirements.txt](file:///d:/Projects/AutoViz_AI/requirements.txt)**: Python package list.
- **[datasets/](file:///d:/Projects/AutoViz_AI/datasets)**: Preloaded sample CSV files (Sales, Student Scores, Demographics) for immediate testing.

---

## 🚀 Setup & Running Guide

Ensure Python 3.10+ is installed on your local environment.

### 1. Set Up the Virtual Environment
Create the virtual environment folder:
```powershell
python -m venv .venv
```

### 2. Activate and Install Dependencies
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

## 💡 Prompt Guidelines

Try entering these query strings into the AI prompt field:
- `"show sales trend over time"` (pre-configures a Line chart mapping Month to Sales)
- `"distribution of age data"` (pre-configures a Histogram showing the spread)
- `"compare student marks vs study hours"` (pre-configures a Scatter plot mapping StudyHours to Marks)
- `"composition of sales"` (pre-configures a Pie chart)
- `"3d scatter student marks vs study hours vs student name"` (pre-configures a 3D Scatter plot mapping X, Y, and Z axes)
