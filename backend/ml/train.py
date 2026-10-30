import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .data import get_dataset_split, FEATURE_METADATA, FEATURE_IDS
from .evaluate import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

MODEL_CONFIGS = {
    "knn": {
        "display_name": "K-Nearest Neighbors (KNN)",
        "file_name": "knn_model.joblib",
        "description": "Instance-based learning with StandardScaler and distance weighting (k=5).",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
        ])
    },
    "random_forest": {
        "display_name": "Random Forest",
        "file_name": "random_forest_model.joblib",
        "description": "Ensemble of 100 trees with balanced class weighting for default detection.",
        "factory": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "file_name": "decision_tree_model.joblib",
        "description": "Interpretable tree model with max_depth=5 and balanced class weighting.",
        "factory": lambda: DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        )
    }
}


def ensure_models_dir():
    """Ensure the backend/models directory exists."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def get_feature_importances(model: Any, model_key: str) -> Any:
    """
    Extracts feature importances if the model supports it (Tree-based).
    KNN does not have built-in feature importances.
    """
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        feature_scores = []
        for feat, score in zip(FEATURE_METADATA, raw_importances):
            feature_scores.append({
                "feature_id": feat["id"],
                "feature_name": feat["name"],
                "label": feat["label"],
                "label_uz": feat["label_uz"],
                "importance": round(float(score), 4),
                "importance_pct": round(float(score) * 100, 2)
            })
        # Sort descending by importance
        feature_scores.sort(key=lambda x: x["importance"], reverse=True)
        return feature_scores
    elif model_key == "knn":
        return {
            "available": False,
            "explanation": "Standard KNN does not provide built-in feature importance because it is an instance-based lazy learner based on spatial Euclidean distance.",
            "explanation_uz": "Oddiy KNN algoritmi masofaga asoslangan bo'lgani sababli ichki xususiyat muhimligi (feature importance) ko'rsatkichiga ega emas."
        }
    return None


def train_and_save_all_models() -> Dict[str, Any]:
    """
    Trains KNN, Random Forest, and Decision Tree on the exact UCI dataset split.
    Evaluates each model, saves the fitted model pipeline/estimator to .joblib,
    and updates metadata.json.
    """
    ensure_models_dir()

    X_train, X_test, y_train, y_test = get_dataset_split()

    trained_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "training_date": trained_date,
        "dataset_info": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_count": int(X_train.shape[1]),
            "feature_ids": FEATURE_IDS
        },
        "models": {}
    }

    best_f1 = -1.0
    best_model_name = ""

    for key, config in MODEL_CONFIGS.items():
        # Initialize model / pipeline
        model = config["factory"]()

        # Measure training time
        t_start_train = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t_start_train

        # Measure prediction time
        t_start_pred = time.time()
        eval_metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            training_time=training_time,
            prediction_time=0.0
        )
        prediction_time = time.time() - t_start_pred
        eval_metrics["prediction_time"] = round(prediction_time, 4)

        # Feature importances
        importances = get_feature_importances(model, key)

        model_metadata = {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "file_name": config["file_name"],
            "trained_at": trained_date,
            "metrics": eval_metrics,
            "feature_importances": importances,
            "feature_names": FEATURE_IDS
        }

        # Save model using joblib
        model_path = os.path.join(MODELS_DIR, config["file_name"])
        joblib.dump(model, model_path)

        # Check best model by F1-score
        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            best_model_name = config["display_name"]

        results["models"][key] = model_metadata

    results["best_model_by_f1"] = {
        "model_name": best_model_name,
        "f1_score": best_f1
    }

    # Save metadata JSON for instant retrieval without reloading full joblib files
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def load_saved_model(model_key: str) -> Any:
    """
    Safely loads a trained model from backend/models/ by key.
    Rejects any unverified keys to prevent arbitrary file loading.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model key '{model_key}'. Allowed: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]
    model_path = os.path.join(MODELS_DIR, config["file_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The '{config['display_name']}' has not been trained yet. Please train models first using the Train button."
        )

    return joblib.load(model_path)


def get_saved_models_metadata() -> Dict[str, Any]:
    """
    Returns the saved metadata of trained models.
    """
    if not os.path.exists(METADATA_FILE):
        return {
            "trained": False,
            "message": "No models have been trained yet. Please click 'Train All Models'.",
            "models": {}
        }

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["trained"] = True
            return data
    except Exception:
        return {
            "trained": False,
            "message": "Model metadata could not be loaded. Please re-train models.",
            "models": {}
        }
