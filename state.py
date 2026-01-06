"""
State definition for the Agentic MLOps Pipeline.
Defines the shared state that flows between agents in the LangGraph workflow.
"""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    Shared state for the multi-agent MLOps pipeline.
    
    Attributes:
        data_path: Path to the input data file (CSV).
        model_path: Path where the trained model is saved.
        model: The trained ML model object.
        metrics: Dictionary containing evaluation metrics (accuracy, mae, etc.).
        feedback: Instructions for backtracking when validation fails.
        messages: Chat history for agent communication.
        status: Current pipeline status (cleaning, training, packaging, error).
        
    Internal fields (for data passing between agents):
        _X_train, _X_test: Feature data for training/testing.
        _y_train, _y_test: Target data for training/testing.
        _feature_columns: List of feature column names.
        _target_column: Name of target column.
        _cleaning_strategy: Strategy used for data cleaning.
    """
    # Public state
    data_path: str
    model_path: str
    model: Any
    metrics: dict[str, float]
    feedback: str
    messages: list[BaseMessage]
    status: str
    
    # Internal state (for passing data between agents)
    _cleaned_data: list[dict]
    _feature_columns: list[str]
    _target_column: str
    _cleaning_strategy: str
    _best_params: dict
    _cv_score: float
    _dockerfile_path: str
    _serve_path: str

