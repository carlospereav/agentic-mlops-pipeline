"""
Machine Learning tools for the MLOps pipeline.
Implements model training, evaluation, and serialization with scikit-learn.
"""

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# Directory for saving models
MODELS_DIR = Path(__file__).parent.parent / "models"


@tool
def train_classifier(
    X_train: list[dict],
    y_train: list,
    model_type: Literal["random_forest", "logistic_regression"] = "random_forest",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a classification model using scikit-learn.
    
    Args:
        X_train: Training features as list of dictionaries.
        y_train: Training labels as list.
        model_type: Type of classifier ('random_forest' or 'logistic_regression').
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary with trained 'model' object and training metadata.
    """
    # Convert to DataFrame
    X = pd.DataFrame(X_train)
    y = pd.Series(y_train)
    
    # Select and configure model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported: 'random_forest', 'logistic_regression'")
    
    # Train the model
    model.fit(X, y)
    
    return {
        "model": model,
        "model_type": model_type,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": list(X.columns),
        "classes": model.classes_.tolist(),
    }


@tool
def evaluate_model(
    model: Any,
    X_test: list[dict],
    y_test: list,
    threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Evaluate a trained model and calculate classification metrics.
    
    Args:
        model: Trained scikit-learn classifier.
        X_test: Test features as list of dictionaries.
        y_test: True labels for test set.
        threshold: Minimum accuracy threshold for approval.
        
    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1),
        approval status, and feedback if rejected.
    """
    # Convert to DataFrame
    X = pd.DataFrame(X_test)
    y_true = pd.Series(y_test)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle binary and multiclass
    average = "binary" if len(set(y_true)) == 2 else "weighted"
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    
    # Decision: approve or reject
    approved = accuracy >= threshold
    
    feedback = ""
    if not approved:
        feedback = (
            f"Model rejected: accuracy ({accuracy:.2%}) below threshold ({threshold:.0%}). "
            f"Suggestions: 1) Check for class imbalance, 2) Normalize numeric features, "
            f"3) Remove outliers, 4) Engineer new features."
        )
    
    return {
        "metrics": metrics,
        "threshold": threshold,
        "approved": approved,
        "feedback": feedback,
        "n_test_samples": len(y_test),
        "predictions_sample": y_pred[:10].tolist(),
    }


@tool
def save_model(
    model: Any,
    model_name: str = "model",
    include_metadata: bool = True,
    metadata: dict | None = None,
) -> dict[str, str]:
    """
    Save a trained model to disk using pickle (.pkl format).
    
    Args:
        model: Trained scikit-learn model to save.
        model_name: Name for the saved model file (without extension).
        include_metadata: Whether to save metadata alongside the model.
        metadata: Optional dictionary with additional metadata to save.
        
    Returns:
        Dictionary with paths to saved model and metadata files.
    """
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the model as .pkl
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    result = {
        "model_path": str(model_path),
        "model_name": model_name,
    }
    
    # Save metadata if requested
    if include_metadata:
        meta = metadata or {}
        meta.update({
            "model_type": type(model).__name__,
            "model_file": f"{model_name}.pkl",
        })
        
        # Add model-specific info if available
        if hasattr(model, "n_features_in_"):
            meta["n_features"] = model.n_features_in_
        if hasattr(model, "classes_"):
            meta["classes"] = model.classes_.tolist()
        if hasattr(model, "feature_names_in_"):
            meta["feature_names"] = model.feature_names_in_.tolist()
        
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        result["metadata_path"] = str(metadata_path)
    
    return result

