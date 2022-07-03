import pandas as pd
import numpy as np
from typing import Callable, Any


def fix_shape(func) -> Callable:
    return_type_hint = func.__annotations__["return"]

    def fix_shape_helper(X1: np.ndarray, x2: np.ndarray) -> return_type_hint:
        if len(x2.shape) == 1:
            x2 = x2.reshape(1, -1)
        else:
            raise Exception("x2 must be array with 1 dimension")

        return func(X1, x2)

    return fix_shape_helper


@fix_shape
def get_euclidean_distance(X1: np.ndarray, x2: np.ndarray) -> np.ndarray[Any, np.dtype[np.float]]:
    return ((X1 - x2) ** 2).sum(axis=1) ** 0.5


@fix_shape
def get_manhattan_distance(X1: np.ndarray, x2: np.ndarray) -> np.ndarray[Any, np.dtype[np.float]]:
    return (X1 - x2).sum(axis=1)


@fix_shape
def get_hamming_distance(X1: np.ndarray, x2: np.ndarray) -> np.ndarray[Any, np.dtype[np.int]]:
    return (X1 != x2).sum(axis=1)


@fix_shape
def get_matching_distance(X1: np.ndarray, x2: np.ndarray) -> np.ndarray[Any, np.dtype[np.int]]:
    return (X1 != x2).mean(axis=1)


def get_distance_matrix(X1: np.ndarray,
                        X2: np.ndarray,
                        distance_function: Callable = get_euclidean_distance) -> pd.DataFrame:
    distance_matrix = []
    for x2 in X2:
        distance_matrix.append(distance_function(X1, x2))

    return pd.DataFrame(distance_matrix).transpose()
