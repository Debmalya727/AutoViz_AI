import pandas as pd
import numpy as np
import os
import re
import json
from dotenv import load_dotenv

# Load environment variables if .env file exists
load_dotenv()

def detect_outliers(df: pd.DataFrame, col: str):
    """Detect outliers in a numeric column using IQR method."""
    if not pd.api.types.is_numeric_dtype(df[col]):
        return pd.DataFrame()
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

def analyze_columns(df: pd.DataFrame):
    """Categorize columns into numeric, categorical, datetime, etc."""
    summary = {}
    for col in df.columns:
        col_type = "categorical"
        
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 5:  # Small unique set is likely categorical/ordinal
                col_type = "categorical"
            else:
                col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
            # Try to convert to datetime to verify
            try:
                pd.to_datetime(df[col], errors='coerce')
                col_type = "datetime"
            except:
                pass
        
        summary[col] = {
            "type": col_type,
            "nullable_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist()
        }
    return summary

def get_llm_query_parsing(query: str, df_summary: str, api_type: str, api_key: str):
    """Use Gemini or OpenAI to parse NL queries to structured configuration."""
    if api_type == "Gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze this natural language query: "{query}"
            And map it to the columns of the following dataset:
            {df_summary}
            
            Return a JSON object only. No markdown formatting (like ```json), just raw JSON.
            JSON structure:
            {{
                "chart_type": "bar" | "line" | "scatter" | "hist" | "pie" | "box",
                "x_column": "exact_column_name_from_dataset",
                "y_column": "exact_column_name_from_dataset" or null,
                "explanation": "Brief explanation of why this chart and these columns were chosen"
            }}
            """
            response = model.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
            return json.loads(text.strip())
        except Exception as e:
            return {"error": str(e)}
            
    elif api_type == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            Analyze this natural language query: "{query}"
            And map it to the columns of the following dataset:
            {df_summary}
            
            Return JSON only.
            JSON structure:
            {{
                "chart_type": "bar" | "line" | "scatter" | "hist" | "pie" | "box",
                "x_column": "exact_column_name_from_dataset",
                "y_column": "exact_column_name_from_dataset" or null,
                "explanation": "Brief explanation of why this chart and these columns were chosen"
            }}
            """
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
            
    return {"error": "Invalid API type"}

def get_llm_insights(df_stats: str, chart_info: str, api_type: str, api_key: str):
    """Use Gemini or OpenAI to generate rich executive insights."""
    if api_type == "Gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Provide a professional business and data analyst executive summary and detailed insights based on the following dataset statistics and generated chart:
            
            Dataset Statistics:
            {df_stats}
            
            Selected Chart Configuration:
            {chart_info}
            
            Format your response in neat markdown. Write 2-3 key findings, possible causes, and actionable recommendations.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error generating LLM insights: {e}"
            
    elif api_type == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            Provide a professional business and data analyst executive summary and detailed insights based on the following dataset statistics and generated chart:
            
            Dataset Statistics:
            {df_stats}
            
            Selected Chart Configuration:
            {chart_info}
            
            Format your response in neat markdown. Write 2-3 key findings, possible causes, and actionable recommendations.
            """
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error generating LLM insights: {e}"
            
    return "API not configured."
