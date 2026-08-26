"""
Data Engineer Agent for the MLOps pipeline.
Responsible for:
- Data cleaning and handling nulls
- Encoding categorical variables (LabelEncoder)
- Data preparation for ML models

NOTE: This agent prepares data. Train/test split is done by Mathematician
to allow for grid search, cross-validation, etc.
"""

from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage
from sklearn.preprocessing import LabelEncoder

from state import AgentState
from tools.data_tools import clean_null_values, load_csv_data


# Available cleaning strategies to try on feedback
CLEANING_STRATEGIES = ["mean", "median", "drop"]


def data_engineer_node(state: AgentState) -> dict[str, Any]:
    """
    Data Engineer agent node - LangGraph compatible.
    
    1. Loads CSV data
    2. Cleans null values
    3. Encodes categorical variables
    4. Passes numeric data to Mathematician
    
    Args:
        state: Current AgentState with data_path and optional feedback.
        
    Returns:
        Updated state with cleaned, encoded data ready for Mathematician.
    """
    messages = list(state.get("messages", []))
    feedback = state.get("feedback", "")
    data_path = state.get("data_path", "")
    
    # Track retry attempts via messages to determine which strategy to use
    retry_count = _count_cleaning_attempts(messages)
    
    try:
        # Determine cleaning strategy based on feedback
        if feedback and retry_count < len(CLEANING_STRATEGIES):
            strategy = CLEANING_STRATEGIES[retry_count]
            action_msg = (
                f"[DataEngineer] Feedback received: '{feedback}'. "
                f"Attempting alternative strategy: '{strategy}' (attempt {retry_count + 1})"
            )
        else:
            strategy = "mean"  # Default strategy
            action_msg = f"[DataEngineer] Starting data processing for: {data_path}"
        
        messages.append(AIMessage(content=action_msg))
        
        # Step 1: Load the CSV data
        load_result = load_csv_data.invoke({"file_path": data_path})
        raw_data = load_result["data"]
        columns = load_result["columns"]
        
        messages.append(AIMessage(
            content=f"[DataEngineer] Loaded {load_result['shape']['rows']} rows, "
                   f"{load_result['shape']['cols']} columns. "
                   f"Null counts: {load_result['null_counts']}"
        ))
        
        # Step 2: Clean null values with selected strategy
        if strategy == "drop":
            cleaned_data = _drop_null_rows(raw_data)
            messages.append(AIMessage(
                content=f"[DataEngineer] Strategy 'drop': Removed rows with null values. "
                       f"Remaining rows: {len(cleaned_data)}"
            ))
        else:
            clean_result = clean_null_values.invoke({
                "data": raw_data,
                "strategy": strategy,
            })
            cleaned_data = clean_result["data"]
            
            if clean_result["columns_cleaned"]:
                messages.append(AIMessage(
                    content=f"[DataEngineer] Strategy '{strategy}': Cleaned nulls in "
                           f"{clean_result['columns_cleaned']}. "
                           f"Nulls filled: {clean_result['nulls_filled']}"
                ))
            else:
                messages.append(AIMessage(
                    content="[DataEngineer] No null values found. Data is clean."
                ))
        
        # Step 3: Detect target column
        target_column = _detect_target_column(columns)
        feature_columns = [c for c in columns if c != target_column]
        
        # Step 4: Encode categorical variables
        df = pd.DataFrame(cleaned_data)
        encoded_data, label_encoders, categorical_cols = _encode_categorical_features(
            df, feature_columns
        )
        
        if categorical_cols:
            messages.append(AIMessage(
                content=f"[DataEngineer] Encoded {len(categorical_cols)} categorical columns: "
                       f"{categorical_cols}"
            ))
        
        messages.append(AIMessage(
            content=f"[DataEngineer] Data ready for Mathematician. "
                   f"Samples: {len(encoded_data)}, "
                   f"Features: {feature_columns}, "
                   f"Target: '{target_column}'"
        ))
        
        # Return cleaned and encoded data
        return {
            "messages": messages,
            "status": "cleaned",
            "feedback": "",
            "_cleaned_data": encoded_data.to_dict(orient="records"),
            "_target_column": target_column,
            "_feature_columns": feature_columns,
            "_cleaning_strategy": strategy,
            "_label_encoders": label_encoders,
            "_categorical_columns": categorical_cols,
        }
        
    except Exception as e:
        error_msg = f"[DataEngineer] Error: {str(e)}"
        messages.append(AIMessage(content=error_msg))
        
        return {
            "messages": messages,
            "status": "error",
            "feedback": error_msg,
        }


def _count_cleaning_attempts(messages: list) -> int:
    """Count how many cleaning attempts have been made based on messages."""
    count = 0
    for msg in messages:
        if hasattr(msg, "content") and "[DataEngineer] Strategy" in msg.content:
            count += 1
    return count


def _detect_target_column(columns: list[str]) -> str:
    """Detect the target column from column names."""
    common_targets = [
        "target", "label", "class", "y", "output", "prediction",
        "exam_score", "score", "price", "salary", "revenue",
    ]
    
    for col in columns:
        if col.lower() in common_targets:
            return col
    
    return columns[-1] if columns else ""


def _drop_null_rows(data: list[dict]) -> list[dict]:
    """Drop rows that contain any null values."""
    df = pd.DataFrame(data)
    df_clean = df.dropna()
    return df_clean.to_dict(orient="records")


def _encode_categorical_features(
    df: pd.DataFrame, 
    feature_columns: list[str]
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame with all data.
        feature_columns: List of feature column names to encode.
        
    Returns:
        Tuple of (encoded DataFrame, dict of encoders, list of categorical columns).
    """
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = []
    
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                categorical_cols.append(col)
    
    return df_encoded, label_encoders, categorical_cols
