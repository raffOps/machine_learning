import numpy as np


def get_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    return (y == predicted).mean()
