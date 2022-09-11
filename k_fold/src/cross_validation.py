import os
import sys
import random
from typing import Any
from time import time
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from Kfold.kfold import Kfold


logger = logging.getLogger()
logger.setLevel("DEBUG")


def get_random_combinations_of_parameters(
        parameters: dict,
        number_of_combinations: int
) -> list[int]:
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
            score = f1_score(y_test_tunning, predictions)
            scores.append(score)
        mean_time = np.asarray(times).mean()
        scores = np.asarray(scores)
        mean_score = scores.mean()
        std_score = scores.std()
        grid_results.append(
            dict(combination, f1_mean_score=mean_score, f1_score_std=std_score, mean_time=mean_time)
        )
    grid_results = pd.DataFrame(grid_results).sort_values(
        by=["f1_mean_score"],
        ascending=False
    ).reset_index(drop=True)

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
    scores = []
    for index_fold, (
            x_train_validation,
            x_test_validation,
            y_train_validation,
            y_test_validation
    ) in enumerate(validation_kfold.split(X, y)):
        logger.info(f"fold: {index_fold}")
        print(f"fold: {index_fold}")
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
        score = f1_score(y_test_validation, predictions)
        scores.append({"classifier": cls, f"tuned_parameters": {str(best_parameters)},   "score": score})

    return pd.DataFrame(scores).sort_values(by="score", ascending=False).reset_index(drop=True)


def main():
    df = pd.read_csv("../data/winequality-red.csv", sep=";")
    y = df.quality.apply(lambda quality: 0 if quality <= 5 else 1).to_numpy()
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1])
    number_of_parameters_combinations = 10
    number_of_folds = 10

    classifier = RandomForestClassifier
    classifier_name = "random_forest"
    parameters_grid = {
        "n_estimators": [50, 100, 200, 300, 500],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": ["sqrt", "log2", 0.2, None]
    }

    results = run_cross_validation(
        classifier=classifier,
        classifier_name=classifier_name,
        parameters_grid=parameters_grid,
        number_of_parameters_combinations=number_of_parameters_combinations,
        numbers_of_folds=number_of_folds,
        values=(X, y)
    )
    with open("../pickle/best_random_forest_classifier", "wb") as fp:
        pickle.dump(results.classifier[0], fp)

    results[["tuned_parameters",
             "score"]].to_csv(f"../data/results/{classifier_name}/validation.csv")


if __name__ == "__main__":
    main()
