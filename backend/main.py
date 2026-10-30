import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ml.data import (
    load_uci_data,
    get_dataset_summary,
    FEATURE_METADATA,
    FEATURE_IDS,
    FEATURE_NAME_MAP,
    NAME_TO_ID_MAP
)
from ml.train import (
    train_and_save_all_models,
    load_saved_model,
    get_saved_models_metadata,
    MODELS_DIR,
    MODEL_CONFIGS
)

app = FastAPI(
    title="Credit Card Default Prediction API",
    description="Full Stack Machine Learning API using UCI Default of Credit Card Clients Dataset (ID: 350)",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Check API service health and runtime status."""
    return {
        "status": "healthy",
        "service": "Credit Card Default Prediction Backend",
        "dataset": "UCI Default of Credit Card Clients (ID: 350)",
        "models_supported": list(MODEL_CONFIGS.keys())
    }


@app.get("/api/dataset/summary", tags=["Dataset"])
def dataset_summary() -> Dict[str, Any]:
    """
    Returns total rows, total features, target distribution,
    missing values, and basic statistics for each feature.
    """
    try:
        return get_dataset_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load dataset summary: {str(e)}"
        )


@app.post("/api/train", tags=["Model Training"])
def train_models() -> Dict[str, Any]:
    """
    Trains all three models (KNN, Random Forest, Decision Tree) on the real UCI dataset.
    Evaluates them on the held-out test set, saves .joblib models and metadata.
    """
    try:
        results = train_and_save_all_models()
        return {
            "status": "success",
            "message": "All 3 models trained and saved successfully.",
            **results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.get("/api/models", tags=["Models"])
def get_models() -> Dict[str, Any]:
    """Returns saved models metadata, evaluation metrics, and training status."""
    return get_saved_models_metadata()


@app.get("/api/models/{model_name}/download", tags=["Models"])
def download_model(model_name: str):
    """
    Downloads the trained model .joblib file for the specified model key:
    knn, random_forest, or decision_tree.
    """
    normalized_key = model_name.lower().strip()
    if normalized_key not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[normalized_key]
    file_path = os.path.join(MODELS_DIR, config["file_name"])

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file '{config['file_name']}' does not exist yet. Please train the models first using the 'Train' button."
        )

    return FileResponse(
        path=file_path,
        filename=config["file_name"],
        media_type="application/octet-stream"
    )


class PredictionInput(BaseModel):
    # Flexible input accepting either X1..X23 or descriptive names
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing values for all 23 features, keyed by ID (X1..X23) or standard name (LIMIT_BAL, etc.)."
    )


@app.post("/api/predict/{model_name}", tags=["Prediction"])
def predict(model_name: str, payload: PredictionInput) -> Dict[str, Any]:
    """
    Makes a prediction for a single client using the chosen saved model.
    Validates all 23 input features and returns prediction, probability,
    and a clear risk explanation.
    """
    normalized_key = model_name.lower().strip()
    if normalized_key not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model_name}'. Allowed models: {list(MODEL_CONFIGS.keys())}"
        )

    # Load model
    try:
        model = load_saved_model(normalized_key)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

    # Validate and build feature array
    input_dict = payload.features
    ordered_values = []
    missing_fields = []
    invalid_fields = []

    for feat in FEATURE_METADATA:
        fid = feat["id"]
        fname = feat["name"]

        # Check if provided via fid or fname
        val = None
        if fid in input_dict:
            val = input_dict[fid]
        elif fname in input_dict:
            val = input_dict[fname]

        if val is None:
            missing_fields.append(f"{fid} ({fname})")
            continue

        try:
            numeric_val = float(val)
            ordered_values.append(numeric_val)
        except (ValueError, TypeError):
            invalid_fields.append(f"{fid} ({fname}): expected numeric value, got '{val}'")

    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required features: {', '.join(missing_fields)}"
        )

    if invalid_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid feature values: {'; '.join(invalid_fields)}"
        )

    # Format into DataFrame with correct column names for sklearn Pipeline/Estimator
    input_df = pd.DataFrame([ordered_values], columns=FEATURE_IDS)

    try:
        prediction = int(model.predict(input_df)[0])

        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            probabilities = {
                "non_default_probability": round(float(probs[0]), 4),
                "default_probability": round(float(probs[1]), 4),
                "default_percentage": round(float(probs[1]) * 100, 2)
            }
        else:
            probabilities = {
                "non_default_probability": 1.0 if prediction == 0 else 0.0,
                "default_probability": 1.0 if prediction == 1 else 0.0,
                "default_percentage": 100.0 if prediction == 1 else 0.0
            }

        def_prob = probabilities.get("default_probability", float(prediction))

        if def_prob < 0.30:
            risk_level = "Low Risk (Past xavf)"
            risk_class = "low"
            explanation = "Mijoz kredit majburiyatlarini o'z vaqtida bajarish ehtimoli yuqori. Default xavfi minimal."
        elif def_prob < 0.60:
            risk_level = "Medium Risk (O'rtacha xavf)"
            risk_class = "medium"
            explanation = "Mijoz to'lov kechikishi ehtimoli o'rtacha. Qo'shimcha monitoring yoki kafillik tavsiya etiladi."
        else:
            risk_level = "High Risk (Yuqori xavf)"
            risk_class = "high"
            explanation = "Mijoz keyingi oy to'lovni kechiktirish (default) xavfi yuqori. Kredit ajratishda ehtiyotkorlik zarur."

        return {
            "model_key": normalized_key,
            "model_name": MODEL_CONFIGS[normalized_key]["display_name"],
            "prediction": prediction,
            "prediction_label": "Default (To'lov kechikishi)" if prediction == 1 else "Non-Default (O'z vaqtida to'lov)",
            "probabilities": probabilities,
            "risk_assessment": {
                "level": risk_level,
                "category": risk_class,
                "explanation_uz": explanation,
                "default_score": round(def_prob * 100, 1)
            },
            "disclaimer": "Eslatma: Ushbu model ta'limiy maqsadlarda yaratilgan bo'lib, rasmiy bank yoki yakuniy kredit qarori o'rnini bosa olmaydi."
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/api/visualizations", tags=["Visualizations"])
def get_visualizations() -> Dict[str, Any]:
    """
    Returns aggregated JSON data for frontend visualization:
    - Target class distribution
    - Feature correlations with default
    - Default vs Non-Default demographic and financial comparisons
    - Feature importances for Random Forest and Decision Tree
    """
    try:
        X, y = load_uci_data()

        # 1. Target Distribution
        non_def_cnt = int((y == 0).sum())
        def_cnt = int((y == 1).sum())
        total = len(y)
        target_dist = [
            {
                "name": "Non-Default (0: To'lagan)",
                "count": non_def_cnt,
                "percentage": round((non_def_cnt / total) * 100, 1),
                "fill": "#2563eb"
            },
            {
                "name": "Default (1: Qaytarmagan)",
                "count": def_cnt,
                "percentage": round((def_cnt / total) * 100, 1),
                "fill": "#dc2626"
            }
        ]

        # 2. Feature Correlations with Target Y
        corrs = []
        for feat in FEATURE_METADATA:
            fid = feat["id"]
            corr_val = float(X[fid].corr(y))
            corrs.append({
                "id": fid,
                "name": feat["name"],
                "label": feat["name"],
                "label_uz": feat["label_uz"],
                "correlation": round(corr_val, 4),
                "abs_correlation": round(abs(corr_val), 4)
            })
        corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        # 3. Default vs Non-default comparison on key metrics
        comparison_metrics = [
            {"id": "X1", "name": "LIMIT_BAL", "label": "Avg Credit Limit ($)"},
            {"id": "X5", "name": "AGE", "label": "Avg Age (Years)"},
            {"id": "X6", "name": "PAY_0", "label": "Sep Payment Delay"},
            {"id": "X12", "name": "BILL_AMT1", "label": "Avg Bill Sep ($)"},
            {"id": "X18", "name": "PAY_AMT1", "label": "Avg Payment Sep ($)"}
        ]
        comparison_data = []
        for m in comparison_metrics:
            fid = m["id"]
            avg_non_def = float(X.loc[y == 0, fid].mean())
            avg_def = float(X.loc[y == 1, fid].mean())
            comparison_data.append({
                "metric": m["name"],
                "label": m["label"],
                "non_default": round(avg_non_def, 2),
                "default": round(avg_def, 2)
            })

        # 4. Feature Importances from saved models
        saved_meta = get_saved_models_metadata()
        rf_importance = []
        dt_importance = []
        if saved_meta.get("trained"):
            rf_info = saved_meta.get("models", {}).get("random_forest", {})
            if isinstance(rf_info.get("feature_importances"), list):
                rf_importance = rf_info["feature_importances"][:10]  # Top 10

            dt_info = saved_meta.get("models", {}).get("decision_tree", {})
            if isinstance(dt_info.get("feature_importances"), list):
                dt_importance = dt_info["feature_importances"][:10]  # Top 10

        return {
            "target_distribution": target_dist,
            "top_correlations": corrs[:10],
            "all_correlations": corrs,
            "feature_comparison": comparison_data,
            "feature_importances": {
                "random_forest": rf_importance,
                "decision_tree": dt_importance,
                "knn": {
                    "available": False,
                    "explanation": "Standard KNN does not provide built-in feature importance because it is an instance-based lazy learner based on spatial Euclidean distance.",
                    "explanation_uz": "Standart KNN algoritmi xotiraga asoslangan (instance-based) masofa o'lchovidan foydalangani sababli ichki xususiyat muhimligi (feature importance) ko'rsatkichiga ega emas."
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating visualizations: {str(e)}"
        )
