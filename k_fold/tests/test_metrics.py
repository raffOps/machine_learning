import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from src.metrics import (
    get_confusion_matrix,
    get_accuracy_score,
    get_precision_score,
    get_recall_score,
    get_f1_score
)


@pytest.fixture
def example_1() -> tuple[np.ndarray, [np.ndarray, np.ndarray]]:
    real = np.asarray([0, 1, 1, 1, 1, 0, 1, 0, 1, 0])
    predicted = np.asarray([0, 1, 0, 1, 1, 1, 1, 0, 1, 1])

    confusion_matrix = get_confusion_matrix(real, predicted)

    return np.asarray(confusion_matrix), (real, predicted)


def test_accuracy_score(example_1):
    confusion_matrix, (real, predicted) = example_1
    assert accuracy_score(real, predicted) == get_accuracy_score(confusion_matrix)


def test_precision_score(example_1):
    confusion_matrix, (real, predicted) = example_1
    assert precision_score(real, predicted) == get_precision_score(confusion_matrix)


def test_recall_score(example_1):
    confusion_matrix, (real, predicted) = example_1
    assert recall_score(real, predicted) == get_recall_score(confusion_matrix)


def test_f1_score(example_1):
    confusion_matrix, (real, predicted) = example_1
    assert f1_score(real, predicted) == get_f1_score(confusion_matrix)