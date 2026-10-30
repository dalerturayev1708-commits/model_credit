import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    training_time: float,
    prediction_time: float
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the test set.
    Calculates binary metrics (positive class = 1: Default) and weighted metrics.
    Generates confusion matrix, ROC-AUC, classification report, and timings.
    """
    y_pred = model.predict(X_test)

    # Class probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        roc_auc = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = None
        roc_auc = None

    # Binary metrics for the positive class (1 = Default)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_default = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall_default = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_default = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Weighted metrics across both classes
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Full classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Default (0)", "Default (1)"],
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision_default, 4),      # Default class (1)
        "recall": round(recall_default, 4),            # Default class (1)
        "f1": round(f1_default, 4),                    # Default class (1)
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "weighted_metrics": {
            "precision": round(precision_weighted, 4),
            "recall": round(recall_weighted, 4),
            "f1": round(f1_weighted, 4)
        },
        "confusion_matrix": cm.tolist(),
        "cm_breakdown": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0
        },
        "training_time": round(training_time, 4),
        "prediction_time": round(prediction_time, 4),
        "classification_report": report_dict,
        "metric_notes": {
            "primary_precision": "Kredit qaytarmaslik (Default=1) bo'yicha hisoblangan Precision",
            "primary_recall": "Kredit qaytarmaslik (Default=1) bo'yicha hisoblangan Recall",
            "primary_f1": "Kredit qaytarmaslik (Default=1) bo'yicha hisoblangan F1-score"
        }
    }
