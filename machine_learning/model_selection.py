import random

import pandas as pd
import numpy as np


def get_balanced_holdout(X: pd.DataFrame,
                         y: pd.Series,
                         test_size: float = 0.2,
                         seed: int = 42) -> tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
    indexes_test_sample = []
    y_copy = pd.Series(np.copy(y))
    test_size_count = 0
    get_false_targets = True
    get_true_targets = True

    random.seed(seed)

    while test_size_count <= y.size * test_size:
        if get_false_targets:
            possible_choices_false_target = y_copy[y_copy == 0].index.tolist()
            if possible_choices_false_target:
                random_index_target_false = np.random.choice(possible_choices_false_target)
                indexes_test_sample.append(random_index_target_false)
                test_size_count += 1
                y_copy = pd.Series.drop(y_copy, random_index_target_false)
            else:
                get_false_targets = False
        else:
            pass

        if get_true_targets:
            possible_choices_true_target = y_copy[y_copy == 1].index.tolist()
            if possible_choices_true_target:
                random_index_target_true = np.random.choice(possible_choices_true_target)
                indexes_test_sample.append(random_index_target_true)
                test_size_count += 1
                y_copy = pd.Series.drop(y_copy, random_index_target_true)
            else:
                get_true_targets = False
        else:
            pass

    X_train = np.take(X, y_copy.index, axis=0).reset_index(drop=True).values
    y_train = np.take(y, y_copy.index, axis=0).reset_index(drop=True).values
    X_test = np.take(X, indexes_test_sample, axis=0).reset_index(drop=True).values
    y_test = np.take(y, indexes_test_sample, axis=0).reset_index(drop=True).values

    return X_train, X_test, y_train, y_test
