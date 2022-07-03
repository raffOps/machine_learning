import numpy as np

from machine_learning import EPSILON


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    max_element = x.max(axis=0)
    min_element = x.min(axis=0)
    return (x - min_element) / ((max_element - min_element) + EPSILON)
