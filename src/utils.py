from typing import Dict, List, Union

from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict[str, Union[str, float]]:
    """
    Evaluate the model using the F1 score and accuracy.
    """
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {
        "f1": f1,
        "accuracy": acc,
        "classification_report": classification_report(y_true, y_pred),
    }
