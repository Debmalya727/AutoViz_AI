import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from chart_engine import auto_detect_chart, generate_plotly_chart, generate_matplotlib_chart, save_chart
from ai_engine import process_query, explain_ai_choice
from insight_engine import generate_insight
from utils import analyze_columns
from ml_engine import run_kmeans, run_pca, run_linear_regression
from data_engine import execute_sql_query

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AutoViz AI - Enterprise Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME CONFIGURATION ----------------
st.sidebar.markdown("""
<div style="padding: 10px 0px; border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom:15px;">
    <h2 style="margin:0; font-size: 20px;">⚙️ Studio Configuration</h2>
</div>
""", unsafe_allow_html=True)

is_dark = st.sidebar.checkbox("🌙 Enable Dark Mode", value=False)
theme_choice = st.sidebar.selectbox("🎨 Color Palette", ["Cool Indigo", "Emerald Garden", "Sunset Orange", "Dark Charcoal"])

# Dynamic Color Schemes mapping
if is_dark:
    t_colors = {
        "primary": {"Cool Indigo": "#6366f1", "Emerald Garden": "#10b981", "Sunset Orange": "#f97316", "Dark Charcoal": "#64748b"}[theme_choice],
        "primary_gradient": {
            "Cool Indigo": "linear-gradient(135deg, #4f46e5, #6366f1)",
            "Emerald Garden": "linear-gradient(135deg, #059669, #10b981)",
            "Sunset Orange": "linear-gradient(135deg, #ea580c, #f97316)",
            "Dark Charcoal": "linear-gradient(135deg, #334155, #475569)"
        }[theme_choice],
        "bg_gradient": "linear-gradient(180deg, #0b0f19 0%, #111827 50%, #0b0f19 100%)",
        "text_color": "#f8fafc",
        "text_muted": "#94a3b8",
        "title_color": "#ffffff",
        "sidebar_bg": "linear-gradient(180deg, #111827 0%, #0b0f19 100%)",
        "border_color": "rgba(255,255,255,0.06)",
        "card_bg": "rgba(30, 41, 59, 0.45)",
        "card_border": "rgba(255, 255, 255, 0.08)",
        "card_shadow": "0 25px 55px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255,255,255,0.1)",
        "metric_bg": "rgba(30, 41, 59, 0.6)",
        "metric_shadow": "0 10px 25px rgba(0, 0, 0, 0.35)",
        "metric_hover_shadow": "0 15px 35px rgba(0, 0, 0, 0.45)",
        "pill_bg": "rgba(30, 41, 59, 0.7)",
        "accent": {"Cool Indigo": "#312e81", "Emerald Garden": "#064e3b", "Sunset Orange": "#7c2d12", "Dark Charcoal": "#1e293b"}[theme_choice],
        "divider_color": "rgba(255, 255, 255, 0.08)"
    }
else:
    bg_gradients = {
        "Cool Indigo": "linear-gradient(180deg, #f8fbff 0%, #eef4ff 45%, #f8fbff 100%)",
        "Emerald Garden": "linear-gradient(180deg, #f5fdf9 0%, #e6f7ee 45%, #f5fdf9 100%)",
        "Sunset Orange": "linear-gradient(180deg, #fffbf5 0%, #ffeedd 45%, #fffbf5 100%)",
        "Dark Charcoal": "linear-gradient(180deg, #fafafa 0%, #f1f5f9 45%, #fafafa 100%)"
    }
    accents = {
        "Cool Indigo": "#dbeafe",
        "Emerald Garden": "#d1fae5",
        "Sunset Orange": "#ffedd5",
        "Dark Charcoal": "#e2e8f0"
    }
    primaries = {
        "Cool Indigo": "#2563eb",
        "Emerald Garden": "#059669",
        "Sunset Orange": "#ea580c",
        "Dark Charcoal": "#1e293b"
    }
    primary_gradients = {
        "Cool Indigo": "linear-gradient(135deg, #2563eb, #4f46e5)",
        "Emerald Garden": "linear-gradient(135deg, #059669, #10b981)",
        "Sunset Orange": "linear-gradient(135deg, #ea580c, #f97316)",
        "Dark Charcoal": "linear-gradient(135deg, #1e293b, #475569)"
    }
    t_colors = {
        "primary": primaries[theme_choice],
        "primary_gradient": primary_gradients[theme_choice],
        "bg_gradient": bg_gradients[theme_choice],
        "text_color": "#0f172a",
        "text_muted": "#475569",
        "title_color": "#0f172a",
        "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)",
        "border_color": "rgba(15, 23, 42, 0.05)",
        "card_bg": "rgba(255, 255, 255, 0.8)",
        "card_border": "rgba(255, 255, 255, 0.7)",
        "card_shadow": "0 20px 50px rgba(15, 23, 42, 0.06), inset 0 1px 0 rgba(255,255,255,0.8)",
        "metric_bg": "linear-gradient(145deg, #ffffff, #f8fafc)",
        "metric_shadow": "0 10px 25px rgba(15, 23, 42, 0.03)",
        "metric_hover_shadow": "0 15px 35px rgba(15, 23, 42, 0.07)",
        "pill_bg": "rgba(255, 255, 255, 0.75)",
        "accent": accents[theme_choice],
        "divider_color": "rgba(148, 163, 184, 0.18)"
    }

# Inject premium custom styling with neumorphic properties
st.markdown(f"""
<style>
/* ========== GLOBAL ========== */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {t_colors["bg_gradient"]};
    color: {t_colors["text_color"]};
    transition: all 0.3s ease;
}}

/* Hide Streamlit default headers & footers */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Block spacing */
.block-container {{
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

/* Sidebar styling override */
[data-testid="stSidebar"] {{
    background: {t_colors["sidebar_bg"]};
    border-right: 1px solid {t_colors["border_color"]};
    padding-top: 1rem;
}}

[data-testid="stSidebar"] * {{
    color: {t_colors["text_color"]} !important;
}}

/* Neumorphic glassmorphic cards */
.sidebar-box {{
    background: {t_colors["card_bg"]};
    border: 1px solid {t_colors["card_border"]};
    border-radius: 18px;
    padding: 16px;
    box-shadow: {t_colors["card_shadow"]};
    margin-bottom: 16px;
}}

.section-box {{
    background: {t_colors["card_bg"]};
    border: 1px solid {t_colors["card_border"]};
    border-radius: 28px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: {t_colors["card_shadow"]};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transform-style: preserve-3d;
    transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s ease;
}}

.section-box:hover {{
    transform: translateY(-3px);
    box-shadow: 0 30px 65px rgba(0,0,0,{"0.3" if is_dark else "0.08"});
}}

/* Metric cards with 3D hover effects */
.metric-card {{
    background: {t_colors["metric_bg"]};
    border: 1px solid {t_colors["card_border"]};
    border-radius: 22px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: {t_colors["metric_shadow"]};
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}}

.metric-card:hover {{
    transform: translateY(-6px) scale(1.03);
    box-shadow: {t_colors["metric_hover_shadow"]};
}}

.metric-value {{
    font-size: 36px;
    font-weight: 800;
    color: {t_colors["primary"]};
    margin-bottom: 6px;
    line-height: 1.1;
    text-shadow: 0 4px 10px rgba(0,0,0,0.05);
}}

.metric-label {{
    font-size: 14px;
    font-weight: 600;
    color: {t_colors["text_muted"]};
}}

/* Hero banner */
.hero-wrap {{
    background: {t_colors["card_bg"]};
    border: 1px solid {t_colors["card_border"]};
    border-radius: 34px;
    padding: 42px 48px;
    box-shadow: {t_colors["card_shadow"]};
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}}

.hero-wrap::before {{
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -40px;
    top: -40px;
    background: radial-gradient(circle, {t_colors["primary"]}25, transparent 70%);
    border-radius: 50%;
}}

.badge {{
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: {t_colors["accent"]};
    color: {t_colors["primary"]};
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 16px;
    box-shadow: 0 6px 15px {t_colors["primary"]}22;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

.hero-title {{
    font-size: 54px;
    font-weight: 900;
    line-height: 1.1;
    color: {t_colors["title_color"]};
    margin-bottom: 12px;
    letter-spacing: -1px;
}}

.hero-sub {{
    font-size: 18px;
    color: {t_colors["text_muted"]};
    max-width: 850px;
    line-height: 1.6;
    font-weight: 400;
}}

.pill-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}}

.feature-pill {{
    background: {t_colors["pill_bg"]};
    border: 1px solid {t_colors["card_border"]};
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    color: {t_colors["text_color"]};
}}

/* Form inputs & dropdowns */
[data-testid="stWidgetLabel"] p {{
    color: {t_colors["title_color"]} !important;
    font-weight: 700 !important;
}}

[data-testid="stTextInput"] > div > div, 
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {{
    background-color: {"#1e293b" if is_dark else "#ffffff"} !important;
    border: 1px solid {"rgba(255,255,255,0.12)" if is_dark else "#cbd5e1"} !important;
    border-radius: 12px !important;
}}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] * {{
    color: {t_colors["text_color"]} !important;
    -webkit-text-fill-color: {t_colors["text_color"]} !important;
}}

ul[data-baseweb="menu"] {{
    background-color: {"#1e293b" if is_dark else "#ffffff"} !important;
    border: 1px solid {"rgba(255,255,255,0.15)" if is_dark else "#cbd5e1"} !important;
    border-radius: 12px !important;
}}
ul[data-baseweb="menu"] li {{
    color: {t_colors["text_color"]} !important;
    background-color: transparent !important;
}}
ul[data-baseweb="menu"] li:hover {{
    background-color: {t_colors["accent"]} !important;
    color: {t_colors["primary"]} !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background-color: {"#1e293b55" if is_dark else "#ffffff"} !important;
    border: 2px dashed {"rgba(255,255,255,0.2)" if is_dark else "#94a3b8"} !important;
    border-radius: 18px !important;
}}
[data-testid="stFileUploaderDropzone"] section > div > span {{
    color: {t_colors["text_color"]} !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background: {"#334155" if is_dark else "#f8fafc"} !important;
    color: {t_colors["primary"]} !important;
    border: 1px solid {"rgba(255,255,255,0.1)" if is_dark else "#cbd5e1"} !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}}

/* Buttons */
.stDownloadButton button,
.stButton button {{
    background: {t_colors["primary_gradient"]} !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.35rem !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 25px {t_colors["primary"]}44 !important;
    transition: all 0.22s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}}

.stDownloadButton button:hover,
.stButton button:hover {{
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 15px 35px {t_colors["primary"]}66 !important;
}}

/* Headings */
h1, h2, h3, h4, h5, h6 {{
    color: {t_colors["title_color"]} !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
}}

/* Dividers & lines */
.premium-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {t_colors["divider_color"]}, transparent);
    margin: 24px 0;
}}

.soft-note {{
    font-size: 14px;
    color: {t_colors["text_muted"]};
    margin-top: -6px;
    margin-bottom: 16px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.markdown("""
<div class="sidebar-box">
    <strong style="font-size:16px;">📈 Visual Properties</strong>
</div>
""", unsafe_allow_html=True)

chart_mode = st.sidebar.radio("Plotting Method", ["Interactive (Plotly 3D)", "Static (Matplotlib 3D)"], label_visibility="collapsed")
render_mode = "Plotly" if "Interactive" in chart_mode else "Matplotlib"

st.sidebar.markdown("""
<div class="sidebar-box" style="margin-top:15px;">
    <strong style="font-size:16px;">🤖 API Integration</strong>
</div>
""", unsafe_allow_html=True)

api_provider = st.sidebar.selectbox("Model Provider", ["None", "Gemini", "OpenAI"])
api_key = ""
if api_provider != "None":
    api_key = st.sidebar.text_input(f"{api_provider} API Key", type="password", placeholder=f"Enter {api_provider} key")

if st.sidebar.button("🧹 Reset Workspace"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------- HERO HEADER ----------------
st.markdown(f"""
<div class="hero-wrap">
    <div class="badge">AutoViz AI • Enterprise ML Studio</div>
    <div class="hero-title">AutoViz AI Studio</div>
    <div class="hero-sub">
        Slice databases using SQLite commands, model interactions in 3D coordinate charts,
        run unsupervised K-Means clusters, and render dimensionality PCA scatter projections.
    </div>
    <div class="pill-row">
        <div class="feature-pill">🧬 Theme: {theme_choice}</div>
        <div class="feature-pill">🕹 Engine: {chart_mode}</div>
        <div class="feature-pill">🔑 LLM Bridge: {api_provider}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- DATA LOADER ----------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📂 Set Input Dataset Source")
st.SoftNote = st.markdown('<div class="soft-note">Load pre-packaged analytics sets or drag custom CSV/Excel spreadsheets.</div>', unsafe_allow_html=True)

dataset_option = st.selectbox(
    "Select dataset source",
    [
        "Sample: Sales Trends (Monthly)",
        "Sample: Student Scores (Performance)",
        "Sample: Age Demographics (Distribution)",
        "Upload custom file..."
    ],
    label_visibility="collapsed"
)

df_raw = None

if dataset_option == "Upload custom file...":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    sample_file_map = {
        "Sample: Sales Trends (Monthly)": "datasets/sales.csv",
        "Sample: Student Scores (Performance)": "datasets/students.csv",
        "Sample: Age Demographics (Distribution)": "datasets/age_data.csv"
    }
    sample_path = sample_file_map[dataset_option]
    if os.path.exists(sample_path):
        df_raw = pd.read_csv(sample_path)
    else:
        st.error(f"Sample file not found at {sample_path}")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- MAIN APPLICATION FLOW ----------------
if df_raw is not None:
    # Use session state to manage active dataset (can be queried via SQL)
    if "df_active" not in st.session_state:
        st.session_state.df_active = df_raw.copy()
        st.session_state.df_source_name = dataset_option

    # If the raw dataset changes, reload it automatically
    if st.session_state.df_source_name != dataset_option:
        st.session_state.df_active = df_raw.copy()
        st.session_state.df_source_name = dataset_option
        # clear SQL query
        if "sql_query_str" in st.session_state:
            del st.session_state["sql_query_str"]

    df = st.session_state.df_active

    # Column categorizations
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    all_cols = df.columns.tolist()

    # ---------------- TABS WORKSPACE ----------------
    tab_viz, tab_ml, tab_sql = st.tabs([
        "📊 3D & 2D Visual Canvas", 
        "🤖 Machine Learning Studio", 
        "⚡ SQL Query Editor"
    ])

    # ==========================================
    # TAB 3: SQL QUERY EDITOR (DATA TRANSFORMATION)
    # ==========================================
    with tab_sql:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("⚡ SQLite Database Console")
        st.markdown('<div class="soft-note">Write standard SQL SELECT queries on your active dataset. The dataset table is named <code>data</code>.</div>', unsafe_allow_html=True)

        # Default query template
        default_query = "SELECT * FROM data"
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            default_query = f"SELECT * FROM data ORDER BY {numeric_cols[0]} DESC"

        query_value = st.text_area("SQL Statement", value=st.session_state.get("sql_query_str", default_query), height=110)
        
        c_sql_btn1, c_sql_btn2 = st.columns([0.2, 0.8])
        with c_sql_btn1:
            run_sql = st.button("Run SQL Command")
        with c_sql_btn2:
            if st.button("Reset SQL Filter"):
                st.session_state.df_active = df_raw.copy()
                st.session_state.sql_query_str = default_query
                st.success("Dataset reset to original source data.")
                st.rerun()

        if run_sql:
            st.session_state.sql_query_str = query_value
            result_df, err = execute_sql_query(df_raw, query_value)
            if err:
                st.error(f"SQL Error: {err}")
            else:
                st.session_state.df_active = result_df
                st.success(f"SQL Executed successfully! Loaded **{len(result_df)}** records matching query.")
                df = result_df
                # Recalculate columns
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
                all_cols = df.columns.tolist()
                
        # Show SQL cheatsheet expander
        with st.expander("💡 SQLite Cheat Sheet Examples"):
            st.markdown(f"""
            - **Filter rows**: `SELECT * FROM data WHERE {numeric_cols[0] if numeric_cols else 'column'} > 1000`
            - **Limit output**: `SELECT * FROM data LIMIT 5`
            - **Order items**: `SELECT * FROM data ORDER BY {numeric_cols[0] if numeric_cols else 'column'} DESC`
            - **Simple Aggregations**: `SELECT {categorical_cols[0] if categorical_cols else 'category'}, AVG({numeric_cols[0] if numeric_cols else 'column'}) as Average FROM data GROUP BY {categorical_cols[0] if categorical_cols else 'category'}`
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)


    # ==========================================
    # TAB 1: VISUALIZATION CANVAS
    # ==========================================
    with tab_viz:
        # --- Metadata summary cards ---
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("📌 Active Data Profile")
        st.markdown('<div class="soft-note">Visualizing a slice of the dataset columns.</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]}</div><div class="metric-label">Active Records</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Active Columns</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(numeric_cols)}</div><div class="metric-label">Numeric Vector Columns</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(categorical_cols)}</div><div class="metric-label">Categorical Columns</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Dataset Preview tables
        col_left, col_right = st.columns([1.35, 1])
        with col_left:
            st.markdown('<div class="section-box" style="height: 100%;">', unsafe_allow_html=True)
            st.subheader("📄 Dataset Preview")
            st.markdown('<div class="soft-note">Tabular matrix slice of active dataframe.</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=250)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-box" style="height: 100%;">', unsafe_allow_html=True)
            st.subheader("📋 Structural Attributes")
            st.markdown('<div class="soft-note">Active columns and types profiling.</div>', unsafe_allow_html=True)
            meta = analyze_columns(df)
            meta_recs = [{"Attribute": k, "Data Type": v["type"], "Unique Vals": v["unique_values"]} for k, v in meta.items()]
            st.dataframe(pd.DataFrame(meta_recs), use_container_width=True, height=210, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # Initialize default chart values
        if "chart_type" not in st.session_state:
            st.session_state.chart_type = "Auto"
        if "x_col" not in st.session_state:
            st.session_state.x_col = all_cols[0]
        if "y_col" not in st.session_state:
            st.session_state.y_col = "None"
        if "z_col" not in st.session_state:
            st.session_state.z_col = "None"
        if "ai_explanation" not in st.session_state:
            st.session_state.ai_explanation = "Ready. Input query to automatically set chart configurations."
        if "last_query" not in st.session_state:
            st.session_state.last_query = ""

        # AI playground vs parameters
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("🤖 Interactive AI Canvas Controls")
        st.markdown('<div class="soft-note">Synthesize visualizations using prompts, or customize manual dimensions.</div>', unsafe_allow_html=True)

        c_play_left, c_play_right = st.columns([1.3, 1])

        with c_play_left:
            user_query = st.text_input(
                "Natural Language Request",
                placeholder="e.g. show sales trend over time / distribution of marks / 3d scatter x vs y vs z",
                key="nlp_query_viz_str"
            )
            
            if user_query and user_query != st.session_state.last_query:
                st.session_state.last_query = user_query
                with st.spinner("AI analyzing coordinates..."):
                    chart_type, x_col, y_col, explanation = process_query(user_query, df, api_provider, api_key)
                    
                    # Sync to state
                    st.session_state.chart_type = chart_type
                    st.session_state.x_col = x_col
                    st.session_state.y_col = y_col if y_col else "None"
                    st.session_state.ai_explanation = explanation
                    
                    # Auto handle Z axis if query requests 3D
                    if "3d" in user_query or "3-d" in user_query:
                        if len(numeric_cols) >= 3:
                            st.session_state.chart_type = "scatter_3d"
                            st.session_state.z_col = numeric_cols[2]
                        elif len(numeric_cols) == 2:
                            st.session_state.chart_type = "scatter_3d"
                            st.session_state.z_col = numeric_cols[1]
            
            st.markdown("#### 🧠 AI Heuristic Reason")
            st.info(st.session_state.ai_explanation)

        with c_play_right:
            st.markdown("#### 🛠 Design Specifications")
            
            chart_list = ["Auto", "bar", "line", "scatter", "hist", "pie", "box", "scatter_3d", "line_3d"]
            ct_val = st.session_state.chart_type
            if ct_val not in chart_list:
                ct_val = "Auto"
            ct_idx = chart_list.index(ct_val)
            manual_ct = st.selectbox("Visualization Template", chart_list, index=ct_idx, key="m_ct_sb")
            
            x_idx = all_cols.index(st.session_state.x_col) if st.session_state.x_col in all_cols else 0
            manual_x = st.selectbox("X-Axis dimension", all_cols, index=x_idx, key="m_x_sb")
            
            y_options = ["None"] + numeric_cols + [c for c in categorical_cols if c != manual_x]
            y_options = list(dict.fromkeys(y_options))
            y_val = st.session_state.y_col
            if y_val not in y_options:
                y_val = "None"
            y_idx = y_options.index(y_val)
            manual_y = st.selectbox("Y-Axis dimension", y_options, index=y_idx, key="m_y_sb")
            
            # Z Axis dropdown (Active only when 3D chart is selected)
            is_3d_selected = "3d" in manual_ct or manual_ct == "Auto" and len(numeric_cols) >= 3
            z_options = ["None"] + numeric_cols
            z_options = list(dict.fromkeys(z_options))
            z_val = st.session_state.z_col
            if z_val not in z_options:
                z_val = "None"
            z_idx = z_options.index(z_val)
            
            manual_z = st.selectbox(
                "Z-Axis dimension (3D Charts only)",
                z_options,
                index=z_idx,
                disabled=not is_3d_selected,
                key="m_z_sb"
            )

            # Sync overrides to state
            st.session_state.chart_type = manual_ct
            st.session_state.x_col = manual_x
            st.session_state.y_col = manual_y
            st.session_state.z_col = manual_z

        st.markdown('</div>', unsafe_allow_html=True)

        # Coordinate resolution
        f_chart_type = st.session_state.chart_type
        f_x = st.session_state.x_col
        f_y = None if st.session_state.y_col == "None" else st.session_state.y_col
        f_z = None if st.session_state.z_col == "None" else st.session_state.z_col

        if f_chart_type == "Auto":
            det_ct, det_x, det_y, det_z = auto_detect_chart(df)
            f_chart_type = det_ct
            f_x = det_x
            f_y = det_y
            f_z = det_z

        # Ensure Z is configured if 3D is selected
        if "3d" in f_chart_type and not f_z:
            available_z = [c for c in numeric_cols if c != f_x and c != f_y]
            f_z = available_z[0] if available_z else (numeric_cols[0] if numeric_cols else all_cols[0])

        # --- Graph Canvas Card ---
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("📈 Visualization Canvas")
        st.markdown(f'<div class="soft-note">Displaying rotatable 3D or 2D coordinates for model <b>{f_chart_type.upper()}</b></div>', unsafe_allow_html=True)

        fig_plotly = generate_plotly_chart(df, f_chart_type, f_x, f_y, theme_choice, is_dark, f_z)
        fig_matplotlib = generate_matplotlib_chart(df, f_chart_type, f_x, f_y, theme_choice, is_dark, f_z)

        if render_mode == "Plotly":
            st.plotly_chart(fig_plotly, use_container_width=True)
            html_p = save_chart(fig_plotly, render_mode="Plotly")
            with open(html_p, "r", encoding="utf-8") as file:
                st.download_button(
                    label="⬇&nbsp; Download Interactive HTML Canvas",
                    data=file.read(),
                    file_name="canvas_interactive.html",
                    mime="text/html"
                )
        else:
            st.pyplot(fig_matplotlib, use_container_width=True)
            png_p = save_chart(fig_matplotlib, render_mode="Matplotlib")
            with open(png_p, "rb") as file:
                st.download_button(
                    label="⬇&nbsp; Download Static PNG Frame",
                    data=file.read(),
                    file_name="canvas_static.png",
                    mime="image/png"
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Insight Engine Card ---
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("🧠 Analytical Insight Report")
        st.markdown('<div class="soft-note">Synthesised narrative analysis computed from data dimensions.</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing vectors..."):
            ins_md = generate_insight(df, f_x, f_y, api_provider, api_key, f_chart_type)
            # Add Z axis note if active
            if f_z:
                ins_md += f"\n\n- **3D Coordinate Note**: Z-axis mapped to **{f_z}** to analyze multivariate relationships."
            st.markdown(ins_md)
        st.markdown('</div>', unsafe_allow_html=True)


    # ==========================================
    # TAB 2: MACHINE LEARNING STUDIO
    # ==========================================
    with tab_ml:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("🤖 Machine Learning Analytics Studio")
        st.markdown('<div class="soft-note">Run statistical ML models directly on your active dataset.</div>', unsafe_allow_html=True)

        ml_mode = st.radio("Select ML Analytics Model", ["K-Means Clustering", "PCA 3D Projection", "Linear Regression Fit"], horizontal=True)
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # 1. K-MEANS CLUSTERING
        if ml_mode == "K-Means Clustering":
            st.markdown("### 🤖 Unsupervised K-Means Clustering")
            st.markdown("Group data rows into K clusters based on similarity of numeric values.")
            
            if len(numeric_cols) < 2:
                st.warning("⚠️ K-Means clustering requires at least 2 numerical columns.")
            else:
                features_selected = st.multiselect("Select numerical features for similarity clustering", numeric_cols, default=numeric_cols[:2])
                k_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=3)
                
                if st.button("Calculate Clusters"):
                    with st.spinner("Executing cluster fit..."):
                        clustered_df, km_summary = run_kmeans(df, features_selected, k_clusters)
                        
                        # Plot clusters
                        st.markdown(km_summary)
                        
                        # Graph clusters
                        if len(features_selected) >= 3:
                            # 3D plot
                            st.markdown("#### 🕹 Interactive 3D Clusters Plot")
                            fig_km = px.scatter_3d(
                                clustered_df, 
                                x=features_selected[0], 
                                y=features_selected[1], 
                                z=features_selected[2],
                                color="Cluster",
                                title=f"K-Means Cluster Space (K={k_clusters})",
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig_km.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc" if is_dark else "#0f172a",
                                scene=dict(
                                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
                                )
                            )
                            st.plotly_chart(fig_km, use_container_width=True)
                        else:
                            # 2D plot
                            st.markdown("#### 📊 Interactive 2D Clusters Plot")
                            fig_km = px.scatter(
                                clustered_df, 
                                x=features_selected[0], 
                                y=features_selected[1], 
                                color="Cluster",
                                title=f"K-Means Cluster Space (K={k_clusters})",
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig_km.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc" if is_dark else "#0f172a"
                            )
                            st.plotly_chart(fig_km, use_container_width=True)

        # 2. PCA 3D PROJECTION
        elif ml_mode == "PCA 3D Projection":
            st.markdown("### 🧬 PCA Dimensionality Reduction Projection")
            st.markdown("Reduce high-dimensional datasets into 3 Principal Components (PC1, PC2, PC3) to visualize multivariate structures.")
            
            if len(numeric_cols) < 3:
                st.warning("⚠️ PCA 3D Projection requires a minimum of 3 numerical columns.")
            else:
                pca_features = st.multiselect("Select numerical features to include in reduction", numeric_cols, default=numeric_cols)
                
                # Option to color PCA by some target
                color_by = st.selectbox("Color nodes by column", ["None"] + all_cols)
                
                if st.button("Generate PCA 3D space"):
                    with st.spinner("Fitting principal components..."):
                        pca_df, pca_summary = run_pca(df, pca_features)
                        
                        if pca_df.empty:
                            st.error(pca_summary)
                        else:
                            st.markdown(pca_summary)
                            
                            st.markdown("#### 🧬 PCA Principal Components Space (3D)")
                            
                            color_col = None if color_by == "None" else color_by
                            
                            fig_pca = px.scatter_3d(
                                pca_df,
                                x="PC1",
                                y="PC2",
                                z="PC3",
                                color=color_col,
                                title="3D Principal Components Projection Space",
                                color_continuous_scale="Viridis",
                                color_discrete_sequence=px.colors.qualitative.Safe
                            )
                            fig_pca.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc" if is_dark else "#0f172a",
                                scene=dict(
                                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
                                )
                            )
                            st.plotly_chart(fig_pca, use_container_width=True)

        # 3. LINEAR REGRESSION
        elif ml_mode == "Linear Regression Fit":
            st.markdown("### 📈 Linear Regression Line Fitting")
            st.markdown("Fit a linear regression trend line to predict a dependent variable from an independent variable.")
            
            if len(numeric_cols) < 2:
                st.warning("⚠️ Linear regression fitting requires at least 2 numerical columns.")
            else:
                c_reg1, c_reg2 = st.columns(2)
                with c_reg1:
                    reg_x = st.selectbox("Independent Variable (X-axis)", numeric_cols)
                with c_reg2:
                    reg_y = st.selectbox("Dependent Variable (Y-axis)", [c for c in numeric_cols if c != reg_x])
                    
                if st.button("Fit Regression Line"):
                    with st.spinner("Executing regression calculations..."):
                        predictions, clean_idx, reg_summary = run_linear_regression(df, reg_x, reg_y)
                        
                        if predictions is None:
                            st.error(reg_summary)
                        else:
                            st.markdown(reg_summary)
                            
                            # Create plot of data and line
                            clean_df = df.loc[clean_idx].copy()
                            clean_df['Regression Fit'] = predictions
                            
                            fig_reg = go.Figure()
                            # Scatter dots
                            fig_reg.add_trace(go.Scatter(
                                x=clean_df[reg_x], 
                                y=clean_df[reg_y],
                                mode='markers',
                                name='Data Points',
                                marker=dict(size=9, color=t_colors["primary"], opacity=0.7)
                            ))
                            # Line fit
                            fig_reg.add_trace(go.Scatter(
                                x=clean_df[reg_x],
                                y=clean_df['Regression Fit'],
                                mode='lines',
                                name='Fit Line',
                                line=dict(color='red', width=3)
                            ))
                            
                            fig_reg.update_layout(
                                title=f"Regression Fit: {reg_y} predicted by {reg_x}",
                                xaxis_title=reg_x,
                                yaxis_title=reg_y,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#f8fafc" if is_dark else "#0f172a",
                                font_family="'Inter', sans-serif"
                            )
                            st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Showcase Welcome Guide if no dataset loaded
    st.markdown(f"""
    <div class="section-box" style="text-align: center; padding: 50px;">
        <h2 style="margin-bottom: 12px;">📊 Start Your Data Exploration Journey</h2>
        <p style="font-size: 16px; color: {t_colors["text_muted"]}; max-width: 600px; margin: 0 auto 30px auto;">
            Choose a sample dataset from the dropdown above or upload your own CSV/Excel file in the main panel to unlock interactive charts and statistical insight reviews.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="background: {t_colors["metric_bg"]}; border: 1px solid {t_colors["card_border"]}; padding: 20px; border-radius: 16px; width: 180px;">
                <span style="font-size: 28px;">⚡</span>
                <h4 style="margin: 10px 0 5px 0;">SQL Workspace</h4>
                <p style="font-size: 12px; color: {t_colors["text_muted"]}; margin: 0;">Query tabular data with standard SQLite syntax.</p>
            </div>
            <div style="background: {t_colors["metric_bg"]}; border: 1px solid {t_colors["card_border"]}; padding: 20px; border-radius: 16px; width: 180px;">
                <span style="font-size: 28px;">🕹</span>
                <h4 style="margin: 10px 0 5px 0;">3D Canvas</h4>
                <p style="font-size: 12px; color: {t_colors["text_muted"]}; margin: 0;">Visualize rotatable, interactive 3D plots.</p>
            </div>
            <div style="background: {t_colors["metric_bg"]}; border: 1px solid {t_colors["card_border"]}; padding: 20px; border-radius: 16px; width: 180px;">
                <span style="font-size: 28px;">🤖</span>
                <h4 style="margin: 10px 0 5px 0;">ML Studio</h4>
                <p style="font-size: 12px; color: {t_colors["text_muted"]}; margin: 0;">Fit K-Means clusters and PCA dimension projections.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)