"""
Model Evaluation Module:
Calculates Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Loss metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List
from backend.database import save_model_metrics

def compute_model_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    model_name: str,
    val_loss: float = 0.25,
    epochs_trained: int = 15
) -> Dict[str, Any]:
    """Computes full evaluation metrics suite and saves to SQLite database."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average='macro', zero_division=0
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))

    # Save summary to SQLite
    save_model_metrics(
        model_name=model_name,
        accuracy=round(acc, 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1_score=round(float(f1), 4),
        loss=round(float(val_loss), 4),
        epochs_trained=epochs_trained
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "loss": round(float(val_loss), 4),
        "epochs_trained": epochs_trained,
        "class_names": class_names,
        "confusion_matrix": cm.tolist()
    }
