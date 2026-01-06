"""
Mathematician Agent (The Validator) for the MLOps pipeline.
Responsible for model training and validation.

NOTE: This agent only calculates metrics and saves the model.
It does NOT make flow decisions - that's handled by conditional edges in main.py.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from langchain_core.messages import AIMessage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from state import AgentState


# Directory for saving models
MODELS_DIR = Path(__file__).parent.parent / "models"


def mathematician_node(state: AgentState) -> dict[str, Any]:
    """
    Mathematician agent node - LangGraph compatible.
    
    Trains a RandomForest model and calculates validation metrics.
    Saves the trained model as .pkl file.
    
    IMPORTANT: This agent only updates metrics and model state.
    Flow decisions (approve/reject) are made by conditional edges.
    
    Args:
        state: Current AgentState with training data from Data Engineer.
        
    Returns:
        Updated state with model, metrics, and model_path.
    """
    messages = list(state.get("messages", []))
    
    try:
        # Retrieve training data from state
        X_train = state.get("_X_train", [])
        X_test = state.get("_X_test", [])
        y_train = state.get("_y_train", [])
        y_test = state.get("_y_test", [])
        target_column = state.get("_target_column", "target")
        
        if not X_train or not y_train:
            raise ValueError("No training data available. Data Engineer must run first.")
        
        messages.append(AIMessage(
            content=f"[Mathematician] Received {len(X_train)} training samples, "
                   f"{len(X_test)} test samples. Starting model training..."
        ))
        
        # Convert to DataFrames
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # Train RandomForest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_df, y_train_series)
        
        messages.append(AIMessage(
            content=f"[Mathematician] RandomForest trained. "
                   f"Features: {X_train_df.shape[1]}, "
                   f"Classes: {model.classes_.tolist()}"
        ))
        
        # Calculate predictions
        y_pred = model.predict(X_test_df)
        
        # Calculate metrics
        metrics = _calculate_metrics(y_test_series, y_pred)
        
        messages.append(AIMessage(
            content=f"[Mathematician] Validation metrics: "
                   f"Accuracy={metrics['accuracy']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}"
        ))
        
        # Save model as .pkl
        model_path = _save_model_pkl(model, "trained_model", metrics)
        
        messages.append(AIMessage(
            content=f"[Mathematician] Model saved to: {model_path}"
        ))
        
        # Return updated state - NO flow decisions here
        return {
            "messages": messages,
            "model": model,
            "model_path": model_path,
            "metrics": metrics,
            "status": "trained",
            # Preserve internal data for potential retry
            "_X_train": X_train,
            "_X_test": X_test,
            "_y_train": y_train,
            "_y_test": y_test,
            "_target_column": target_column,
        }
        
    except Exception as e:
        error_msg = f"[Mathematician] Error during training: {str(e)}"
        messages.append(AIMessage(content=error_msg))
        
        return {
            "messages": messages,
            "status": "error",
            "feedback": error_msg,
            "metrics": {},
        }


def _calculate_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with accuracy, precision, recall, f1_score, and mae.
    """
    # Determine if binary or multiclass
    n_classes = len(set(y_true))
    average = "binary" if n_classes == 2 else "weighted"
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Calculate MAE for regression-like interpretation
    # Convert to numeric if possible for MAE calculation
    try:
        mae = mean_absolute_error(
            pd.to_numeric(y_true, errors="coerce").fillna(0),
            pd.to_numeric(pd.Series(y_pred), errors="coerce").fillna(0)
        )
    except Exception:
        mae = 0.0
    
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "mae": round(float(mae), 4),
    }


def _save_model_pkl(model: Any, model_name: str, metrics: dict) -> str:
    """
    Save model to .pkl file with metadata.
    
    Args:
        model: Trained scikit-learn model.
        model_name: Base name for the model file.
        metrics: Dictionary of model metrics.
        
    Returns:
        Path to saved model file.
    """
    import json
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model as .pkl
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "model_file": f"{model_name}.pkl",
        "metrics": metrics,
    }
    
    if hasattr(model, "n_features_in_"):
        metadata["n_features"] = int(model.n_features_in_)
    if hasattr(model, "classes_"):
        metadata["classes"] = model.classes_.tolist()
    if hasattr(model, "feature_names_in_"):
        metadata["feature_names"] = model.feature_names_in_.tolist()
    
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return str(model_path)
