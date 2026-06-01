import pandas as pd
import numpy as np
from utils import detect_outliers, get_llm_insights

def generate_insight(df: pd.DataFrame, x: str, y: str, api_type: str = None, api_key: str = None, chart_type: str = "bar"):
    """
    Generate deep analytical insights about the data.
    Uses LLM if API key is provided, otherwise generates a rich local statistical report.
    """
    try:
        # 1. Gather all local statistical information
        stats_summary = []
        chart_info = f"Chart Type: {chart_type}, X-axis: {x}, Y-axis: {y if y else 'None'}"
        
        # Safe column retrieval
        if x not in df.columns:
            return "⚠️ Selected columns do not exist in the dataset."
        if y and y not in df.columns:
            y = None

        if y is None:
            # Univariate analysis
            col_data = df[x].dropna()
            if pd.api.types.is_numeric_dtype(df[x]):
                mean = col_data.mean()
                median = col_data.median()
                std = col_data.std()
                min_v = col_data.min()
                max_v = col_data.max()
                outliers = detect_outliers(df, x)
                
                stats_summary.append(f"Univariate analysis on numeric column '{x}':")
                stats_summary.append(f"- Mean: {mean:.4f}")
                stats_summary.append(f"- Median: {median:.4f}")
                stats_summary.append(f"- Standard Deviation: {std:.4f}")
                stats_summary.append(f"- Range: [{min_v}, {max_v}]")
                stats_summary.append(f"- Total rows: {len(col_data)}")
                stats_summary.append(f"- Outliers count: {len(outliers)}")
                
                # Format local markdown
                local_markdown = (
                    f"### 📊 Distribution Analysis: **{x}**\n\n"
                    f"Here is the local statistical profile of the distribution:\n\n"
                    f"- **Average (Mean)**: `{mean:.2f}`\n"
                    f"- **Middle Value (Median)**: `{median:.2f}`\n"
                    f"- **Spread (Std Dev)**: `{std:.2f}` (indicates {'high' if std > (mean * 0.5) else 'low'} variance)\n"
                    f"- **Range**: `{min_v}` to `{max_v}`\n"
                    f"- **Data Completeness**: `{len(col_data) / len(df) * 100:.1f}%` ({len(col_data)} of {len(df)} rows)\n"
                )
                if len(outliers) > 0:
                    local_markdown += f"- **⚠️ Anomalies (Outliers)**: `{len(outliers)}` rows lie outside the 1.5x IQR boundary."
                else:
                    local_markdown += "- **Anomalies**: No extreme statistical outliers detected."
            else:
                # Categorical
                counts = col_data.value_counts()
                unique_c = len(counts)
                top_cat = counts.index[0] if unique_c > 0 else "N/A"
                top_val = counts.values[0] if unique_c > 0 else 0
                pct = (top_val / len(col_data)) * 100 if len(col_data) > 0 else 0
                
                stats_summary.append(f"Univariate analysis on categorical column '{x}':")
                stats_summary.append(f"- Unique values: {unique_c}")
                stats_summary.append(f"- Most common: '{top_cat}' (frequency: {top_val}, {pct:.2f}%)")
                stats_summary.append(f"- Top categories breakdown: {counts.head(5).to_dict()}")
                
                local_markdown = (
                    f"### 🗂 Categorical Summary: **{x}**\n\n"
                    f"- **Cardinality**: `{unique_c}` unique categories detected.\n"
                    f"- **Dominant Class**: **{top_cat}** represents `{pct:.1f}%` of the entries ({top_val} occurrences).\n"
                    f"- **Top Categories Frequency**:\n"
                )
                for cat, val in counts.head(5).items():
                    local_markdown += f"  - **{cat}**: {val} rows ({val / len(col_data) * 100:.1f}%)\n"
                    
        else:
            # Bivariate analysis
            col_x = df[x]
            col_y = df[y].dropna()
            
            y_mean = col_y.mean()
            y_median = col_y.median()
            y_std = col_y.std()
            y_min = col_y.min()
            y_max = col_y.max()
            
            # Find context rows for max/min
            max_mask = df[y] == y_max
            min_mask = df[y] == y_min
            
            max_x_val = df[max_mask][x].dropna().values[0] if any(max_mask) else "N/A"
            min_x_val = df[min_mask][x].dropna().values[0] if any(min_mask) else "N/A"
            
            outliers = detect_outliers(df, y)
            
            stats_summary.append(f"Bivariate analysis '{y}' grouped/mapped by '{x}':")
            stats_summary.append(f"- {y} statistics: Mean={y_mean:.4f}, Median={y_median:.4f}, Std={y_std:.4f}, Range=[{y_min}, {y_max}]")
            stats_summary.append(f"- Peak: {y_max} at {x}='{max_x_val}'")
            stats_summary.append(f"- Lowest: {y_min} at {x}='{min_x_val}'")
            
            # Trend calculation
            trend_desc = ""
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                corr = df[x].corr(df[y])
                stats_summary.append(f"- Linear Correlation (Pearson): {corr:.4f}")
                if corr > 0.7:
                    trend_desc = f"📈 **Strong positive correlation** (`r = {corr:.2f}`). As **{x}** increases, **{y}** climbs significantly."
                elif corr > 0.3:
                    trend_desc = f"📈 **Moderate positive correlation** (`r = {corr:.2f}`). There is a general upward trend between **{x}** and **{y}**."
                elif corr < -0.7:
                    trend_desc = f"📉 **Strong negative correlation** (`r = {corr:.2f}`). As **{x}** increases, **{y}** decreases significantly."
                elif corr < -0.3:
                    trend_desc = f"📉 **Moderate negative correlation** (`r = {corr:.2f}`). There is a general downward trend between **{x}** and **{y}**."
                else:
                    trend_desc = f"⚖️ **No strong linear correlation** (`r = {corr:.2f}`). The relationship between **{x}** and **{y}** appears weak or non-linear."
            else:
                # If X is categorical/temporal and we are plotting line/bar, check sequence direction
                # Sort values by index to preserve natural chronological order (like months or dates)
                try:
                    sorted_df = df.dropna(subset=[x, y])
                    if len(sorted_df) > 1:
                        first_y = sorted_df[y].iloc[0]
                        last_y = sorted_df[y].iloc[-1]
                        diff = last_y - first_y
                        pct_change = (diff / first_y * 100) if first_y != 0 else 0
                        stats_summary.append(f"- Directional sequence difference: {diff:.4f} ({pct_change:.2f}%)")
                        if diff > 0:
                            trend_desc = f"📈 **Upward Trajectory**: Net growth of **+{diff:,.2f}** ({pct_change:+.1f}%) observed from start (**{sorted_df[x].iloc[0]}**) to end (**{sorted_df[x].iloc[-1]}**)."
                        elif diff < 0:
                            trend_desc = f"📉 **Downward Trajectory**: Net decline of **{diff:,.2f}** ({pct_change:.1f}%) observed from start (**{sorted_df[x].iloc[0]}**) to end (**{sorted_df[x].iloc[-1]}**)."
                        else:
                            trend_desc = f"⚖️ **Neutral Trajectory**: The values remain steady from start (**{sorted_df[x].iloc[0]}**) to end (**{sorted_df[x].iloc[-1]}**)."
                except Exception as ex:
                    trend_desc = "🔄 Categorical timeline sequence detected without numeric sorting."
            
            local_markdown = (
                f"### 💡 Local Analytical Insights: **{y}** by **{x}**\n\n"
                f"Here is the local analytical summary of the chart data:\n\n"
                f"1. **Peak Performance (Maximum)**:\n"
                f"   - **Highest value**: `{y_max:,.2f}` occurs when **{x}** is **{max_x_val}**.\n\n"
                f"2. **Trough Performance (Minimum)**:\n"
                f"   - **Lowest value**: `{y_min:,.2f}` occurs when **{x}** is **{min_x_val}**.\n\n"
                f"3. **Aggregates & Spread**:\n"
                f"   - **Average (Mean) {y}**: `{y_mean:,.2f}`\n"
                f"   - **Median {y}**: `{y_median:,.2f}`\n"
                f"   - **Standard Deviation**: `{y_std:,.2f}` (measures the variance from average)\n\n"
                f"4. **Trend Profile**:\n"
                f"   - {trend_desc}\n"
            )
            if len(outliers) > 0:
                local_markdown += f"\n5. **⚠️ Anomalies (Outliers)**:\n   - Detected `{len(outliers)}` statistical outliers in **{y}** beyond the standard IQR threshold."
                
        # 2. Check if LLM API is requested and use it, otherwise return local Markdown
        if api_key and api_type:
            stats_str = "\n".join(stats_summary)
            llm_result = get_llm_insights(stats_str, chart_info, api_type, api_key)
            if not llm_result.startswith("⚠️"):
                return llm_result
            else:
                return local_markdown + f"\n\n*(Note: LLM Generation failed: {llm_result.replace('⚠️', '').strip()})*"
                
        return local_markdown

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Insight generation encountered an error: {e}"