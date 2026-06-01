import difflib
import re
import pandas as pd
from utils import get_llm_query_parsing

def process_query(query: str, df: pd.DataFrame, api_type: str = None, api_key: str = None):
    """
    Process the user query and return chart type, X-axis, Y-axis, and an explanation.
    Tries LLM first if API key is provided, falls back to enhanced local rules.
    """
    if api_key and api_type:
        # Prepare a small dataset summary for the LLM
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
        sample_data = df.head(3).to_dict(orient='records')
        
        df_summary = (
            f"Columns: {list(df.columns)}\n"
            f"Numeric Columns: {numeric_cols}\n"
            f"Categorical Columns: {categorical_cols}\n"
            f"Sample Data (first 3 rows): {sample_data}\n"
        )
        
        res = get_llm_query_parsing(query, df_summary, api_type, api_key)
        if "error" not in res and "chart_type" in res:
            x_col = res["x_column"]
            y_col = res["y_column"]
            chart_type = res["chart_type"]
            explanation = res.get("explanation", "AI analyzed the query using LLM.")
            
            # Verify columns exist in dataframe, case-insensitive mapping
            col_map = {c.lower(): c for c in df.columns}
            if x_col and x_col.lower() in col_map:
                x_col = col_map[x_col.lower()]
            else:
                x_col = df.columns[0]
                
            if y_col and y_col.lower() in col_map:
                y_col = col_map[y_col.lower()]
            elif y_col:
                y_col = None
                
            return chart_type, x_col, y_col, explanation

    # Fallback to local rule-based matching
    return process_query_locally(query, df)


def process_query_locally(query: str, df: pd.DataFrame):
    """
    Local rules to extract chart type, match columns using substring & close matches,
    and intelligently assign X and Y axes based on column types and query syntax.
    """
    clean_query = query.lower().strip()
    all_cols = df.columns.tolist()
    lower_cols = [c.lower() for c in all_cols]
    
    # 1. Match Columns (scoring based on query context)
    matched_cols = []
    
    # Direct substring matches
    for idx, col_lower in enumerate(lower_cols):
        # Exact column in query
        if col_lower in clean_query:
            matched_cols.append((all_cols[idx], 10))
        # Word overlap
        else:
            col_words = re.findall(r'\w+', col_lower)
            query_words = re.findall(r'\w+', clean_query)
            overlap = set(col_words).intersection(set(query_words))
            if overlap:
                matched_cols.append((all_cols[idx], len(overlap) * 5))
            else:
                # Fuzzy match words
                for w in query_words:
                    close = difflib.get_close_matches(w, [col_lower], n=1, cutoff=0.7)
                    if close:
                        matched_cols.append((all_cols[idx], 4))
                        break
                        
    # Sort matched columns by score descending, remove duplicates while preserving order
    seen = set()
    unique_matched = []
    for col, score in sorted(matched_cols, key=lambda x: x[1], reverse=True):
        if col not in seen:
            seen.add(col)
            unique_matched.append(col)
            
    # 2. Intent detection for Chart Type
    chart_type = "bar"
    if any(k in clean_query for k in ["trend", "over time", "temporal", "growth", "line"]):
        chart_type = "line"
    elif any(k in clean_query for k in ["relationship", "versus", " vs ", "correlation", "scatter", "compare two"]):
        chart_type = "scatter"
    elif any(k in clean_query for k in ["distribution", "spread", "range", "histogram", "hist"]):
        chart_type = "hist"
    elif any(k in clean_query for k in ["composition", "pie", "share", "proportion", "percentage", "donut"]):
        chart_type = "pie"
    elif any(k in clean_query for k in ["box", "outlier", "quantile", "percentile"]):
        chart_type = "box"
    elif any(k in clean_query for k in ["compare", "comparison", "bar", "by"]):
        chart_type = "bar"

    # 3. Determine X and Y columns
    x_col = None
    y_col = None
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    if chart_type == "hist":
        # Histogram needs 1 column, preferably numeric
        if unique_matched:
            # Prefer first matched numeric
            num_matched = [c for c in unique_matched if c in numeric_cols]
            x_col = num_matched[0] if num_matched else unique_matched[0]
        else:
            x_col = numeric_cols[0] if numeric_cols else all_cols[0]
        y_col = None
    else:
        # Needs 2 columns (X and Y)
        if len(unique_matched) >= 2:
            col1, col2 = unique_matched[0], unique_matched[1]
            
            # Syntax check: check if 'by' or 'over' precedes a matched column in the query
            by_matches = re.findall(r'(?:by|over|vs)\s+(\w+)', clean_query)
            if by_matches:
                by_col_name = by_matches[0]
                close_by_col = difflib.get_close_matches(by_col_name, lower_cols, n=1, cutoff=0.6)
                if close_by_col:
                    target_col = all_cols[lower_cols.index(close_by_col[0])]
                    if target_col in [col1, col2]:
                        x_col = target_col
                        y_col = col2 if x_col == col1 else col1
            
            # Type-based check if axis order is not clear from syntax
            if not x_col:
                # If one is categorical/date and other numeric, categorical is X, numeric is Y
                if col1 in numeric_cols and col2 not in numeric_cols:
                    x_col = col2
                    y_col = col1
                elif col2 in numeric_cols and col1 not in numeric_cols:
                    x_col = col1
                    y_col = col2
                else:
                    # Default: first matched is X, second is Y
                    x_col = col1
                    y_col = col2
        elif len(unique_matched) == 1:
            col = unique_matched[0]
            # Match second column from the remaining dataset columns
            other_cols = [c for c in all_cols if c != col]
            if col in numeric_cols:
                # If matched is numeric, try to find a categorical/date column for X, and use matched as Y
                cat_others = [c for c in other_cols if c not in numeric_cols]
                if cat_others:
                    x_col = cat_others[0]
                    y_col = col
                else:
                    x_col = other_cols[0] if other_cols else col
                    y_col = col
            else:
                # Matched is categorical/date, find a numeric column for Y
                num_others = [c for c in other_cols if c in numeric_cols]
                x_col = col
                y_col = num_others[0] if num_others else (other_cols[0] if other_cols else None)
        else:
            # Zero matched columns, use defaults
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                if chart_type not in ["scatter", "line"]:
                    chart_type = "scatter"
            elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                x_col, y_col = categorical_cols[0], numeric_cols[0]
            else:
                x_col = all_cols[0]
                y_col = all_cols[1] if len(all_cols) > 1 else None

    # Sanity checks
    if not x_col:
        x_col = all_cols[0]
    if y_col == x_col:
        y_col = None

    explanation = (
        f"Detected intent for a **{chart_type.upper()}** chart. "
        f"Matched query keywords to columns: **{x_col}** (X-axis) and **{y_col if y_col else 'None'}** (Y-axis)."
    )
    
    return chart_type, x_col, y_col, explanation


def explain_ai_choice(query: str, chart_type: str, x_col: str, y_col: str):
    """Return a detailed text explaining why the chart was selected."""
    return (
        f"Based on the analysis of your request: **'{query}'**:\n\n"
        f"- **Chart Type**: Selected **{chart_type.upper()}** because the request signals "
        f"{'a trend analysis' if chart_type == 'line' else 'a comparative analysis' if chart_type == 'bar' else 'a relationship correlation' if chart_type == 'scatter' else 'a distribution analysis' if chart_type == 'hist' else 'a composition breakdown' if chart_type == 'pie' else 'a distribution spread comparison'}.\n"
        f"- **X-Axis / Independent Variable**: **{x_col}** matches the key attribute or category.\n"
        f"- **Y-Axis / Dependent Variable**: **{y_col if y_col else 'No numeric target column (univariate)'}** is measured."
    )