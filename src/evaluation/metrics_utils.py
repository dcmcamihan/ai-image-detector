import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, average: str = "binary", as_dataframe: bool = False, y_score=None):
    """
    Compute common classification metrics.

    Args:
        y_true: list or tensor of ground-truth labels
        y_pred: list or tensor of predicted labels
        average: 'binary' for 2-class, 'macro' for multi-class

    Returns:
        dict of accuracy, precision, recall, f1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    # Optional ROC-AUC for binary classification when scores provided
    if y_score is not None and average == "binary":
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.detach().cpu().numpy()
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            result["roc_auc"] = float("nan")
    if as_dataframe:
        return pd.DataFrame([result])
    return result


