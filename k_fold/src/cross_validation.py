import random
from typing import Any
from time import time
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from kfold import Kfold
from metrics import (
    get_confusion_matrix,
    get_f1_score
)


def get_random_combinations_of_parameters(
        parameters: dict,
        number_of_combinations: int
) -> list[dict[Any, Any]]:
    combinations = []
    for _ in range(number_of_combinations):
        combination = {}
        for parameter, values in parameters.items():
            combination[parameter] = random.choice(values)
        combinations.append(combination)
    return combinations


def get_best_parameters(
        classifier: Any,
        parameters_grid: dict[str, list[Any]],
        numbers_of_folds: int,
        number_of_parameters_combinations: int,
        values: tuple[np.ndarray, np.ndarray]
) -> tuple[dict[str, Any], pd.DataFrame]:
    grid_results = []
    parameters_combinations = get_random_combinations_of_parameters(
        parameters_grid,
        number_of_parameters_combinations
    )
    X, y = values
    for combination in parameters_combinations:
        tunning_kfold = Kfold(numbers_of_folds)
        times = []
        scores = []
        for (
                x_train_tunning,
                x_test_tunning,
                y_train_tunning,
                y_test_tunning
        ) in tunning_kfold.split(X, y):
            initial_time = time()
            cls = classifier(**combination)
            cls.fit(x_train_tunning, y_train_tunning)
            final_time = time()
            times.append(final_time - initial_time)
            predictions = cls.predict(x_test_tunning)
            confusion_matrix = get_confusion_matrix(y_test_tunning, predictions)
            score = get_f1_score(confusion_matrix.values)
            scores.append(score)
        mean_time = np.asarray(times).mean()
        scores = np.asarray(scores)
        mean_score = scores.mean()
        std_score = scores.std()
        grid_results.append(
            dict(combination, f1_mean_score=mean_score, f1_std_score=std_score, mean_time=mean_time)
        )
    grid_results = pd.DataFrame(grid_results).\
        sort_values(by=["f1_mean_score"], ascending=False).\
        reset_index(drop=True)

    best_parameters = grid_results.iloc[:, :-3].to_dict("records")[0]

    return best_parameters, grid_results


def run_cross_validation(
        classifier: Any,
        classifier_name: str,
        parameters_grid: dict[str, Any],
        number_of_parameters_combinations: int,
        numbers_of_folds: int,
        values: tuple[np.ndarray, np.ndarray]
) -> pd.DataFrame:
    validation_kfold = Kfold(k=numbers_of_folds)
    X, y = values
    results = []
    for index_fold, (
            x_train_validation,
            x_test_validation,
            y_train_validation,
            y_test_validation
    ) in enumerate(validation_kfold.split(X, y)):
        best_parameters, tunning_results = get_best_parameters(
            classifier=classifier,
            parameters_grid=parameters_grid,
            numbers_of_folds=numbers_of_folds-1,
            number_of_parameters_combinations=number_of_parameters_combinations,
            values=(x_train_validation, y_train_validation)
        )
        tunning_results.to_csv(f"../data/results/{classifier_name}/tunning/fold_{index_fold}.csv")
        print(f"classifier: {classifier_name}\nbest parameters: {best_parameters}")

        cls = classifier(**best_parameters)
        cls.fit(x_train_validation, y_train_validation)
        predictions = cls.predict(x_test_validation)
        confusion_matrix = get_confusion_matrix(y_test_validation, predictions)
        validation_score = get_f1_score(confusion_matrix.values)
        results.append(
            {
                "classifier": cls,
                "tuned_parameters": str(best_parameters),
                "tunning_mean_score": tunning_results.values[0][3],
                "tunning_std_score": tunning_results.values[0][4],
                "validation_score": validation_score
            }
        )

    return pd.DataFrame(results).\
        sort_values(by="validation_score", ascending=False).\
        reset_index(drop=True)
