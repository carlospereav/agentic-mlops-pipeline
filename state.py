"""
State definition for the Agentic MLOps Pipeline.
Defines the shared state that flows between agents in the LangGraph workflow.
"""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
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
    """
    data_path: str
    model_path: str
    model: Any
    metrics: dict[str, float]
    feedback: str
    messages: list[BaseMessage]
    status: str

