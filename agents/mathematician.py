"""
Mathematician Agent (The Validator) for the MLOps pipeline.
Responsible for:
- Train/test split (for grid search flexibility)
- Model training with GridSearchCV
- Rigorous validation

NOTE: This agent controls the split to enable grid search, cross-validation, etc.
Flow decisions are made by conditional edges in main.py, not here.
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
from sklearn.model_selection import GridSearchCV, train_test_split

from state import AgentState


# Directory for saving models
MODELS_DIR = Path(__file__).parent.parent / "models"

# Grid search parameters for RandomForest
RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}


def mathematician_node(state: AgentState) -> dict[str, Any]:
    """
    Mathematician agent node - LangGraph compatible.
    
    1. Receives cleaned data from Data Engineer
    2. Splits into train/test (controls split for grid search)
    3. Trains RandomForest with GridSearchCV
    4. Evaluates and saves model as .pkl
    
    Args:
        state: Current AgentState with cleaned data.
        
    Returns:
        Updated state with model, metrics, and model_path.
    """
    messages = list(state.get("messages", []))
    
    try:
        # Get cleaned data from Data Engineer
        cleaned_data = state.get("_cleaned_data", [])
        target_column = state.get("_target_column", "target")
        feature_columns = state.get("_feature_columns", [])
        
        if not cleaned_data:
            raise ValueError("No cleaned data available. Data Engineer must run first.")
        
        messages.append(AIMessage(
            content=f"[Mathematician] Received {len(cleaned_data)} samples. "
                   f"Starting train/test split and model training..."
        ))
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_data)
        X = df[feature_columns]
        y = df[target_column]
        
        # Step 1: Train/test split (Mathematician controls this)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        messages.append(AIMessage(
            content=f"[Mathematician] Split: Train={len(X_train)}, Test={len(X_test)} "
                   f"(stratified by target)"
        ))
        
        # Step 2: GridSearchCV for hyperparameter tuning
        messages.append(AIMessage(
            content=f"[Mathematician] Running GridSearchCV with params: {list(RF_PARAM_GRID.keys())}"
        ))
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_model,
            RF_PARAM_GRID,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        
        # Best model from grid search
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        messages.append(AIMessage(
            content=f"[Mathematician] GridSearch complete. "
                   f"Best params: {best_params}, CV score: {cv_score:.4f}"
        ))
        
        # Step 3: Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = _calculate_metrics(y_test, y_pred)
        
        messages.append(AIMessage(
            content=f"[Mathematician] Test metrics: "
                   f"Accuracy={metrics['accuracy']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}"
        ))
        
        # Step 4: Save model as .pkl
        model_path = _save_model_pkl(model, "trained_model", metrics, best_params)
        
        messages.append(AIMessage(
            content=f"[Mathematician] Model saved to: {model_path}"
        ))
        
        # Return updated state
        return {
            "messages": messages,
            "model": model,
            "model_path": model_path,
            "metrics": metrics,
            "status": "trained",
            # Preserve data for potential retry
            "_cleaned_data": cleaned_data,
            "_target_column": target_column,
            "_feature_columns": feature_columns,
            "_best_params": best_params,
            "_cv_score": cv_score,
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
    """Calculate classification metrics."""
    n_classes = len(set(y_true))
    average = "binary" if n_classes == 2 else "weighted"
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
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


def _save_model_pkl(
    model: Any, 
    model_name: str, 
    metrics: dict,
    best_params: dict,
) -> str:
    """Save model to .pkl file with metadata."""
    import json
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "model_file": f"{model_name}.pkl",
        "metrics": metrics,
        "best_params": best_params,
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
