import random
from typing import Generator, Set, Tuple, Any
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

random.seed(10)


class Kfold:
    def __init__(self, k: int, shuffle=True):
        self.k = k
        self.shuffle = shuffle
        self.splits = self.generate_splits()
        
    def generate_splits(self) -> set[tuple[int], int]:
        splits = []
        for index_fold_test in range(self.k):
            folds_for_training = []
            splits.append((folds_for_training, index_fold_test))
            for index_fold_train in range(self.k):
                if index_fold_train != index_fold_test:
                    folds_for_training.append(index_fold_train)
    
        splits = set((tuple(folds_for_training), fold_for_test) for folds_for_training, fold_for_test in splits)
        return splits
    
    def generate_folds(self, y_values: np.ndarray[int]) -> list[tuple[int]]:
        indexes_with_false_label = [index for index, value in enumerate(y_values) if value == 0]
        indexes_with_true_label = [index for index, value in enumerate(y_values) if value == 1]
        if self.shuffle:
            random.shuffle(indexes_with_false_label)
            random.shuffle(indexes_with_true_label)
        folds = [[] for _ in range(self.k)]
        for _ in range(int(len(y_values) / self.k)):
            try:
                for fold in folds:
                    index_false_label = indexes_with_false_label.pop(0)
                    index_true_label = indexes_with_true_label.pop(0)
                    fold.extend([index_false_label, index_true_label])
            except IndexError:
                break

        return [tuple(fold) for fold in folds]

    def split(self, X: np.ndarray, y: np.ndarray[int]):
        folds = self.generate_folds(y_values=y)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for train_indexes_folds, test_index_fold in self.splits:
            for train_index in train_indexes_folds:
                X_train.extend(X.take(folds[train_index], axis=0))
                y_train.extend(y.take(folds[train_index], axis=0))
            X_test.extend(X.take(folds[test_index_fold], axis=0))
            y_test.extend(y.take(folds[test_index_fold], axis=0))

            yield X_train, X_test, y_train, y_test


def main():
    df = pd.read_csv("../../data/winequality-red.csv",
                     sep=";")
    y = df.quality.apply(lambda quality: 0 if quality <= 5 else 1).values
    X = MinMaxScaler().fit_transform(df.iloc[:, :-1])

    # for index in [3, 5, 8]:
    #     kfold = Kfold(index)
    #     folds = kfold.generate_folds(y)
    #     with open(f"../../tests/mocks/values_splitted_in_{index}_folds", "wb") as file:
    #         pickle.dump(folds, file)


if __name__ == "__main__":
    main()