import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Define premium color palettes matching the UI
THEME_PALETTES = {
    "Cool Indigo": ["#2563eb", "#4f46e5", "#818cf8", "#a5b4fc", "#c7d2fe"],
    "Emerald Garden": ["#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"],
    "Sunset Orange": ["#ea580c", "#f97316", "#fb923c", "#fdba74", "#fed7aa"],
    "Dark Charcoal": ["#1e293b", "#334155", "#475569", "#64748b", "#94a3b8"]
}

def auto_detect_chart(df):
    """Automatically detect the best chart based on column data types."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    if len(numeric_cols) >= 3:
        return "scatter_3d", numeric_cols[0], numeric_cols[1], numeric_cols[2]
    elif len(numeric_cols) == 2:
        return "scatter", numeric_cols[0], numeric_cols[1], None
    elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        if df[categorical_cols[0]].nunique() <= 10:
            return "bar", categorical_cols[0], numeric_cols[0], None
        else:
            return "line", categorical_cols[0], numeric_cols[0], None
    elif len(numeric_cols) == 1:
        return "hist", numeric_cols[0], None, None
    else:
        return "bar", df.columns[0], df.columns[1] if len(df.columns) > 1 else None, None


def generate_plotly_chart(df, chart_type, x, y, palette_name="Cool Indigo", is_dark=False, z=None):
    """Generate a premium interactive Plotly chart (supports 2D and 3D)."""
    colors = THEME_PALETTES.get(palette_name, THEME_PALETTES["Cool Indigo"])
    
    bg_color = "rgba(0,0,0,0)"
    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(15,23,42,0.06)"
    
    title = f"{chart_type.upper()} Chart: {x}"
    if y:
        title += f" vs {y}"
    if z:
        title += f" vs {z}"
    
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=colors)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True, color_discrete_sequence=colors)
        fig.update_traces(line=dict(width=3.5))
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title, color_discrete_sequence=colors)
        fig.update_traces(marker=dict(size=11, line=dict(width=0.8, color='white')))
    elif chart_type == "hist":
        fig = px.histogram(df, x=x, title=title, color_discrete_sequence=colors, marginal="box")
    elif chart_type == "pie":
        if df[x].nunique() > 10:
            top_cats = df[x].value_counts().index[:9]
            grouped_df = df.copy()
            grouped_df.loc[~grouped_df[x].isin(top_cats), x] = 'Other'
            fig = px.pie(grouped_df, names=x, values=y, title=title, color_discrete_sequence=colors, hole=0.4)
        else:
            fig = px.pie(df, names=x, values=y, title=title, color_discrete_sequence=colors, hole=0.4)
    elif chart_type == "box":
        fig = px.box(df, x=x, y=y, title=title, color_discrete_sequence=colors)
    elif chart_type == "scatter_3d" and y is not None and z is not None:
        fig = px.scatter_3d(df, x=x, y=y, z=z, title=title, color_discrete_sequence=colors)
        fig.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.5, color='white')))
    elif chart_type == "line_3d" and y is not None and z is not None:
        fig = px.line_3d(df, x=x, y=y, z=z, title=title, color_discrete_sequence=colors)
        fig.update_traces(line=dict(width=4.5))
    else:
        # Fallback
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=colors)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=font_color,
        font_family="'Inter', sans-serif",
        title_font_family="'Outfit', 'Inter', sans-serif",
        title_font_size=22,
        title_font_color=font_color,
        title_x=0.0,
        legend_title_font_color=font_color,
        margin=dict(l=50, r=30, t=70, b=50),
        hoverlabel=dict(
            bgcolor="#ffffff" if not is_dark else "#1e293b",
            font_size=13,
            font_family="'Inter', sans-serif"
        )
    )

    if "3d" in chart_type:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    gridcolor=grid_color,
                    backgroundcolor="rgba(0,0,0,0)",
                    color=font_color,
                    title=dict(font=dict(size=12))
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    backgroundcolor="rgba(0,0,0,0)",
                    color=font_color,
                    title=dict(font=dict(size=12))
                ),
                zaxis=dict(
                    gridcolor=grid_color,
                    backgroundcolor="rgba(0,0,0,0)",
                    color=font_color,
                    title=dict(font=dict(size=12))
                )
            )
        )
    else:
        fig.update_layout(
            xaxis=dict(
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                tickfont=dict(size=12),
                title=dict(font=dict(size=14))
            ),
            yaxis=dict(
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                tickfont=dict(size=12),
                title=dict(font=dict(size=14))
            )
        )
    return fig


def generate_matplotlib_chart(df, chart_type, x, y, palette_name="Cool Indigo", is_dark=False, z=None):
    """Generate a clean static Matplotlib chart (supports 2D and 3D)."""
    colors = THEME_PALETTES.get(palette_name, THEME_PALETTES["Cool Indigo"])
    
    if is_dark:
        plt.style.use("dark_background")
        bg_color = "#0f172a"
        text_color = "#f8fafc"
        grid_color = (1.0, 1.0, 1.0, 0.08)
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        bg_color = "#f8fbff"
        text_color = "#0f172a"
        grid_color = "#e2e8f0"

    title = f"{chart_type.upper()} Chart: {x}"
    if y:
        title += f" vs {y}"
    if z:
        title += f" vs {z}"

    # Handle 3D plotting
    if "3d" in chart_type and z is not None:
        fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
        ax = fig.add_subplot(projection='3d')
        ax.set_facecolor(bg_color)
        
        # Draw 3D plots
        if chart_type == "scatter_3d":
            ax.scatter(df[x], df[y], df[z], color=colors[0], s=80, edgecolor=text_color, linewidth=0.5, alpha=0.8)
        elif chart_type == "line_3d":
            ax.plot(df[x], df[y], df[z], color=colors[0], linewidth=3)
            
        ax.set_title(title, fontsize=18, fontweight='bold', pad=18, color=text_color)
        ax.set_xlabel(x, fontsize=12, labelpad=10, color=text_color)
        ax.set_ylabel(y, fontsize=12, labelpad=10, color=text_color)
        ax.set_zlabel(z, fontsize=12, labelpad=10, color=text_color)
        ax.tick_params(colors=text_color, labelsize=10)
        
        # Style 3D grid lines
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(grid_color)
        ax.yaxis.pane.set_edgecolor(grid_color)
        ax.zaxis.pane.set_edgecolor(grid_color)
        
        plt.tight_layout()
        return fig

    # Handle 2D plotting
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    if chart_type == "bar" and y is not None:
        ax.bar(df[x].astype(str), df[y], color=colors[0], edgecolor=text_color, linewidth=0.5, alpha=0.9)
    elif chart_type == "line" and y is not None:
        ax.plot(df[x].astype(str), df[y], marker='o', color=colors[0], linewidth=3, markersize=8)
    elif chart_type == "scatter" and y is not None:
        ax.scatter(df[x], df[y], color=colors[0], s=120, edgecolor=text_color, linewidth=0.5, alpha=0.85)
    elif chart_type == "hist":
        ax.hist(df[x].dropna(), bins=10, color=colors[0], edgecolor=text_color, alpha=0.8)
    elif chart_type == "pie":
        counts = df[x].value_counts() if y is None else df.groupby(x)[y].sum()
        if len(counts) > 8:
            top = counts.head(7)
            other = pd.Series([counts.iloc[7:].sum()], index=['Other'])
            counts = pd.concat([top, other])
        ax.pie(counts, labels=counts.index, colors=colors[:len(counts)], autopct='%1.1f%%', startangle=140,
               wedgeprops=dict(width=0.4, edgecolor=text_color, linewidth=0.5))
    elif chart_type == "box":
        data_to_plot = [df[df[x] == val][y].dropna() for val in df[x].dropna().unique()]
        ax.boxplot(data_to_plot, labels=df[x].dropna().unique(), patch_artist=True,
                   boxprops=dict(facecolor=colors[0], color=text_color),
                   medianprops=dict(color="red", linewidth=1.5))
    else:
        if y is not None:
            ax.bar(df[x].astype(str), df[y], color=colors[0])

    ax.set_title(title, fontsize=18, fontweight='bold', pad=18, color=text_color)
    ax.set_xlabel(x, fontsize=12, labelpad=10, color=text_color)
    if y:
        ax.set_ylabel(y, fontsize=12, labelpad=10, color=text_color)

    ax.tick_params(colors=text_color, labelsize=10)
    plt.xticks(rotation=45, ha='right')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    
    ax.grid(True, linestyle='--', alpha=0.5, color=grid_color)
    
    plt.tight_layout()
    return fig


def save_chart(fig, render_mode="Plotly"):
    """Save chart to the outputs directory."""
    os.makedirs("outputs", exist_ok=True)
    if render_mode == "Plotly":
        path = os.path.join("outputs", "latest_chart.html")
        fig.write_html(path)
    else:
        path = os.path.join("outputs", "latest_chart.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return path