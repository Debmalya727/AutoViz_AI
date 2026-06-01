import pandas as pd
import sqlite3

def execute_sql_query(df: pd.DataFrame, sql_query: str):
    """
    Load a Pandas DataFrame into an in-memory SQLite table named 'data',
    execute the user's SQL query, and return the resulting DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "No dataset loaded."
        
    try:
        # Open an in-memory SQLite connection
        conn = sqlite3.connect(":memory:")
        
        # Load dataframe to SQL table named 'data'
        df.to_sql("data", conn, index=False, if_exists="replace")
        
        # Execute query
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        return result_df, None
    except Exception as e:
        return pd.DataFrame(), str(e)
