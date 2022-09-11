import numpy as np
import pandas as pd

ERROR = 0.0000000001


def get_confusion_matrix(
        real: np.ndarray,
        predicted: np.ndarray
) -> pd.DataFrame:
    confusion_matrix = {
        "predicted_true": {
            "real_true": ((predicted == 1) & (real == 1)).sum(),
            "real_false": ((predicted == 1) & (real == 0)).sum()
        },
        "predicted_false": {
            "real_true": ((predicted == 0) & (real == 1)).sum(),
            "real_false": ((predicted == 0) & (real == 0)).sum()
        }
    }
    return pd.DataFrame(confusion_matrix)


def get_accuracy_score(confusion_matrix: np.ndarray) -> float:
    right_predictions = confusion_matrix.diagonal().sum()
    return right_predictions / confusion_matrix.sum().sum()


def get_precision_score(confusion_matrix: np.ndarray) -> float:
    true_positives = confusion_matrix[0, 0]
    false_positives = confusion_matrix[1, 0]

    return true_positives / (true_positives + false_positives + ERROR)


def get_recall_score(confusion_matrix: np.ndarray) -> float:
    true_positives = confusion_matrix[0, 0]
    false_negatives = confusion_matrix[0, 1]
    return true_positives / (true_positives + false_negatives + ERROR)


def get_f1_score(confusion_matrix: np.ndarray) -> float:
    precision_score = get_precision_score(confusion_matrix)
    recall_score = get_recall_score(confusion_matrix)
    return 2 * (
            (precision_score * recall_score) / (precision_score + recall_score + ERROR)
    )
