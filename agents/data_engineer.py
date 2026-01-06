"""
Data Engineer Agent for the MLOps pipeline.
Responsible for data cleaning, handling nulls, and normalization.
Uses different cleaning strategies when receiving negative feedback.
"""

from typing import Any

from langchain_core.messages import AIMessage

from state import AgentState
from tools.data_tools import clean_null_values, load_csv_data, split_train_test


# Available cleaning strategies to try on feedback
CLEANING_STRATEGIES = ["mean", "median", "drop"]


def data_engineer_node(state: AgentState) -> dict[str, Any]:
    """
    Data Engineer agent node - LangGraph compatible.
    
    Processes raw data, handles nulls, and prepares train/test splits.
    If receives negative feedback, tries a different cleaning strategy.
    
    Args:
        state: Current AgentState with data_path and optional feedback.
        
    Returns:
        Updated state dictionary with cleaned data and status.
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
        
        messages.append(AIMessage(
            content=f"[DataEngineer] Loaded {load_result['shape']['rows']} rows, "
                   f"{load_result['shape']['cols']} columns. "
                   f"Null counts: {load_result['null_counts']}"
        ))
        
        # Step 2: Clean null values with selected strategy
        if strategy == "drop":
            # Drop rows with nulls instead of filling
            cleaned_data = _drop_null_rows(raw_data)
            messages.append(AIMessage(
                content=f"[DataEngineer] Strategy 'drop': Removed rows with null values. "
                       f"Remaining rows: {len(cleaned_data)}"
            ))
        else:
            # Use mean or median strategy
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
                    content=f"[DataEngineer] No null values found. Data is clean."
                ))
        
        # Step 3: Determine target column
        columns = list(cleaned_data[0].keys()) if cleaned_data else []
        target_column = _detect_target_column(columns)
        
        # Step 4: Split into train/test
        split_result = split_train_test.invoke({
            "data": cleaned_data,
            "target_column": target_column,
            "test_size": 0.2,
            "random_state": 42,
        })
        
        messages.append(AIMessage(
            content=f"[DataEngineer] Data split complete. "
                   f"Train: {split_result['split_info']['train_samples']} samples, "
                   f"Test: {split_result['split_info']['test_samples']} samples. "
                   f"Target: '{target_column}', Features: {split_result['feature_columns']}"
        ))
        
        # Return updated state - pass data to next agent
        return {
            "messages": messages,
            "status": "cleaned",
            "feedback": "",  # Clear feedback after processing
            # Internal data for next agent (prefixed with _ to indicate internal use)
            "_X_train": split_result["X_train"],
            "_X_test": split_result["X_test"],
            "_y_train": split_result["y_train"],
            "_y_test": split_result["y_test"],
            "_feature_columns": split_result["feature_columns"],
            "_target_column": split_result["target_column"],
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
    """
    Detect the target column from column names.
    Looks for common target column names, defaults to last column.
    """
    common_targets = ["target", "label", "class", "y", "output", "prediction"]
    
    for col in columns:
        if col.lower() in common_targets:
            return col
    
    # Default to last column
    return columns[-1] if columns else ""


def _drop_null_rows(data: list[dict]) -> list[dict]:
    """Drop rows that contain any null values."""
    import pandas as pd
    df = pd.DataFrame(data)
    df_clean = df.dropna()
    return df_clean.to_dict(orient="records")
