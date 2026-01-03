"""
Data processing tools for the MLOps pipeline.
Implements Pandas-based functions for CSV loading, cleaning, and splitting.
"""

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split

# Security: Define allowed directories for data access
# Modify this list according to your project's trusted data locations
ALLOWED_DATA_DIRECTORIES: list[str] = [
    "data",           # Project data folder
    "datasets",       # Alternative datasets folder
    "input",          # Input data folder
]


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""
    pass


class InvalidFileTypeError(Exception):
    """Raised when an invalid file type is provided."""
    pass


def _validate_file_path(file_path: str, allowed_extensions: list[str] | None = None) -> Path:
    """
    Validate a file path to prevent path traversal attacks.
    
    Args:
        file_path: The file path to validate.
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.parquet']).
        
    Returns:
        Resolved absolute Path object if valid.
        
    Raises:
        PathTraversalError: If the path attempts to access unauthorized directories.
        InvalidFileTypeError: If the file extension is not in the allowed list.
        FileNotFoundError: If the file does not exist.
    """
    # Get project root (parent of tools directory)
    project_root = Path(__file__).parent.parent.resolve()
    
    # Resolve the input path
    input_path = Path(file_path)
    
    # If relative path, resolve against project root
    if not input_path.is_absolute():
        resolved_path = (project_root / input_path).resolve()
    else:
        resolved_path = input_path.resolve()
    
    # Check if path is within allowed directories
    is_allowed = False
    for allowed_dir in ALLOWED_DATA_DIRECTORIES:
        allowed_path = (project_root / allowed_dir).resolve()
        try:
            resolved_path.relative_to(allowed_path)
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise PathTraversalError(
            f"Access denied: Path '{file_path}' is outside allowed directories. "
            f"Allowed directories: {ALLOWED_DATA_DIRECTORIES}"
        )
    
    # Check file extension if restrictions are specified
    if allowed_extensions:
        if resolved_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise InvalidFileTypeError(
                f"Invalid file type: '{resolved_path.suffix}'. "
                f"Allowed types: {allowed_extensions}"
            )
    
    # Check if file exists
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")
    
    # Check if it's actually a file (not a directory)
    if not resolved_path.is_file():
        raise ValueError(f"Path is not a file: {resolved_path}")
    
    return resolved_path


@tool
def load_csv_data(file_path: str) -> dict:
    """
    Load a CSV file into a pandas DataFrame and return it as a dictionary.
    
    Security: This function validates the file path to prevent path traversal
    attacks. Only files within the allowed data directories can be accessed.
    
    Args:
        file_path: Path to the CSV file to load (relative to project root or 
                   within allowed directories).
        
    Returns:
        Dictionary with 'data' containing the DataFrame as dict (records format),
        'columns' with column names, and 'shape' with dimensions.
        
    Raises:
        PathTraversalError: If the path attempts to access unauthorized directories.
        InvalidFileTypeError: If the file is not a CSV file.
        FileNotFoundError: If the file does not exist.
    """
    # Validate and sanitize the file path
    validated_path = _validate_file_path(
        file_path, 
        allowed_extensions=['.csv']
    )
    
    df = pd.read_csv(validated_path)
    
    return {
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns),
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
    }


@tool
def clean_null_values(data: list[dict], strategy: str = "mean") -> dict:
    """
    Clean null values in the dataset by filling with column means.
    
    Args:
        data: List of dictionaries representing the DataFrame records.
        strategy: Strategy for filling nulls. Currently supports 'mean'.
        
    Returns:
        Dictionary with cleaned 'data', 'columns_cleaned' listing affected columns,
        and 'nulls_filled' with count of nulls filled per column.
    """
    df = pd.DataFrame(data)
    
    nulls_before = df.isnull().sum()
    columns_cleaned = []
    nulls_filled = {}
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if strategy == "mean":
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
            columns_cleaned.append(col)
            nulls_filled[col] = int(null_count)
    
    # For non-numeric columns, fill with mode or 'unknown'
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    for col in non_numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()
            fill_value = mode_val[0] if len(mode_val) > 0 else "unknown"
            df[col] = df[col].fillna(fill_value)
            columns_cleaned.append(col)
            nulls_filled[col] = int(null_count)
    
    return {
        "data": df.to_dict(orient="records"),
        "columns_cleaned": columns_cleaned,
        "nulls_filled": nulls_filled,
        "total_nulls_remaining": int(df.isnull().sum().sum()),
    }


@tool
def split_train_test(
    data: list[dict],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split the dataset into training and testing sets.
    
    Args:
        data: List of dictionaries representing the DataFrame records.
        target_column: Name of the target/label column.
        test_size: Proportion of data for testing (default 0.2 = 20%).
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary with 'X_train', 'X_test', 'y_train', 'y_test' as lists,
        plus metadata about the split.
    """
    df = pd.DataFrame(data)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. "
                        f"Available columns: {list(df.columns)}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        "X_train": X_train.to_dict(orient="records"),
        "X_test": X_test.to_dict(orient="records"),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
        "feature_columns": list(X.columns),
        "target_column": target_column,
        "split_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": test_size,
            "random_state": random_state,
        },
    }

