from typing import Any
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from cross_validation import run_cross_validation


def preprocessing_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df.quality.apply(lambda quality: 0 if quality <= 5 else 1).to_numpy()
    x = MinMaxScaler().fit_transform(df.iloc[:, :-1])
    return x, y


def get_models() -> list[tuple[Any, str, dict[str, Any]]]:
    models = [
        (
            RandomForestClassifier,
            "random_forest",
            {
                "n_estimators": [50, 100, 200, 300],
                "criterion": ["gini", "entropy", "log_loss"],
            }
        ),
        (
            LogisticRegression,
            "logistic_regression",
            {
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
                "C": [100, 10, 1.0, 0.1, 0.01]
            },
        ),
        (
            KNeighborsClassifier,
            "k-nearest_neighbors",
            {
                "n_neighbors": [5, 7, 11, 15, 21],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }
        )

    ]
    return models


def run_models_validation(x, y, models, number_of_parameters_combinations, number_of_folds):
    for classifier, classifier_name, parameters_grid in models:
        classifier = classifier
        classifier_name = classifier_name
        parameters_grid = parameters_grid
        results = run_cross_validation(
            classifier=classifier,
            classifier_name=classifier_name,
            parameters_grid=parameters_grid,
            number_of_parameters_combinations=number_of_parameters_combinations,
            numbers_of_folds=number_of_folds,
            values=(x, y)
        )
        with open(f"../pickle/best_{classifier_name}_classifier", "wb") as fp:
            pickle.dump(results.classifier[0], fp)

        results.drop(columns=["classifier"], inplace=True)
    
        results.to_csv(f"../data/results/{classifier_name}/validation.csv")


def main():
    df = pd.read_csv("../data/winequality-red.csv", sep=";")
    x, y = preprocessing_data(df)
    models = get_models()
    number_of_parameters_combinations = 15
    number_of_folds = 30
    run_models_validation(x, y, models, number_of_parameters_combinations, number_of_folds)


if __name__ == "__main__":
    main()
