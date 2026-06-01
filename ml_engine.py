import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int):
    """
    Run K-Means clustering on the selected numeric features.
    Returns a copy of the dataframe with a 'Cluster' column and descriptive statistics.
    """
    if not features or len(df) < n_clusters:
        return df.copy(), "Insufficient data or features selected."
        
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df_clean['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
        
        # Construct summary information
        cluster_summary = f"### 🤖 K-Means Clustering Summary (K={n_clusters})\n\n"
        cluster_summary += f"Clustered `{len(df_clean)}` rows based on features: **{', '.join(features)}**.\n\n"
        for i in range(n_clusters):
            c_size = (df_clean['Cluster'] == str(i)).sum()
            cluster_summary += f"- **Cluster {i}**: {c_size} records ({c_size / len(df_clean) * 100:.1f}%)\n"
            
        return df_clean, cluster_summary
    except Exception as e:
        return df.copy(), f"⚠️ Error running K-Means: {e}"


def run_pca(df: pd.DataFrame, features: list):
    """
    Execute Principal Component Analysis (PCA) to reduce features to 3 dimensions.
    Returns a DataFrame containing PC1, PC2, PC3 and non-numeric columns for styling.
    """
    if len(features) < 3:
        return pd.DataFrame(), "PCA requires at least 3 numerical features."
        
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    try:
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run PCA to 3 components
        pca = PCA(n_components=3, random_state=42)
        pcs = pca.fit_transform(X_scaled)
        
        # Build output dataframe
        pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2', 'PC3'], index=df_clean.index)
        
        # Copy categorical columns for coloring/labelling
        cat_cols = df_clean.select_dtypes(exclude='number').columns.tolist()
        for col in cat_cols:
            pca_df[col] = df_clean[col]
            
        # If there's numeric columns not in features, copy them too for optional coloring
        other_nums = [c for c in df_clean.select_dtypes(include='number').columns if c not in features]
        for col in other_nums:
            pca_df[col] = df_clean[col]
            
        var_ratio = pca.explained_variance_ratio_
        pca_summary = (
            f"### 🧬 PCA Projection Profile\n\n"
            f"Reduced `{len(features)}` dimensions into 3 principal components explaining **{var_ratio.sum() * 100:.2f}%** of total variance:\n"
            f"- **PC1**: explains `{var_ratio[0] * 100:.2f}%` variance\n"
            f"- **PC2**: explains `{var_ratio[1] * 100:.2f}%` variance\n"
            f"- **PC3**: explains `{var_ratio[2] * 100:.2f}%` variance\n"
        )
        return pca_df, pca_summary
    except Exception as e:
        return pd.DataFrame(), f"⚠️ Error executing PCA: {e}"


def run_linear_regression(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Fit linear regression model between x_col and y_col.
    Returns predicted series and a summary statistics string.
    """
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        return None, None, "Linear regression requires numeric X and Y columns."
        
    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    X = df_clean[[x_col]].values
    y = df_clean[y_col].values
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        r2 = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        summary = (
            f"### 📈 Regression Model Fit\n\n"
            f"- **Equation**: `{y_col} = {slope:.4f} * {x_col} + {intercept:.4f}`\n"
            f"- **R² Coefficient**: `{r2:.4f}` (explains `{r2 * 100:.1f}%` of the variance in **{y_col}**)\n"
            f"- **Correlation direction**: {'Positive' if slope > 0 else 'Negative'} slope.\n"
        )
        return predictions, df_clean.index, summary
    except Exception as e:
        return None, None, f"⚠️ Error fitting linear regression: {e}"
