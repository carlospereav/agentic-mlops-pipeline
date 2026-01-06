"""
Data Engineer Agent for the MLOps pipeline.
Responsible for data cleaning, handling nulls, and normalization.
Uses different cleaning strategies when receiving negative feedback.

NOTE: This agent only cleans data. Train/test split is done by Mathematician
to allow for grid search, cross-validation, etc.
"""

from typing import Any

from langchain_core.messages import AIMessage

from state import AgentState
from tools.data_tools import clean_null_values, load_csv_data


# Available cleaning strategies to try on feedback
CLEANING_STRATEGIES = ["mean", "median", "drop"]


def data_engineer_node(state: AgentState) -> dict[str, Any]:
    """
    Data Engineer agent node - LangGraph compatible.
    
    Loads and cleans data, then passes it to Mathematician.
    Does NOT split data - that's the Mathematician's job for grid search flexibility.
    
    Args:
        state: Current AgentState with data_path and optional feedback.
        
    Returns:
        Updated state with cleaned data ready for Mathematician.
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
        
        messages.append(AIMessage(
            content=f"[DataEngineer] Data ready for Mathematician. "
                   f"Samples: {len(cleaned_data)}, "
                   f"Features: {feature_columns}, "
                   f"Target: '{target_column}'"
        ))
        
        # Return cleaned data - Mathematician will do the split
        return {
            "messages": messages,
            "status": "cleaned",
            "feedback": "",
            "_cleaned_data": cleaned_data,
            "_target_column": target_column,
            "_feature_columns": feature_columns,
            "_cleaning_strategy": strategy,
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
    common_targets = ["target", "label", "class", "y", "output", "prediction"]
    
    for col in columns:
        if col.lower() in common_targets:
            return col
    
    return columns[-1] if columns else ""


def _drop_null_rows(data: list[dict]) -> list[dict]:
    """Drop rows that contain any null values."""
    import pandas as pd
    df = pd.DataFrame(data)
    df_clean = df.dropna()
    return df_clean.to_dict(orient="records")
