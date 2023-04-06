import random
import numpy as np

random.seed(42)

BALANCED = False


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
            folds_for_training.extend(
                index_fold_train
                for index_fold_train in range(self.k)
                if index_fold_train != index_fold_test
            )
        return {
            (tuple(folds_for_training), fold_for_test)
            for folds_for_training, fold_for_test in splits
        }
    
    def generate_folds(self, y_values: np.ndarray) -> list[tuple[int]]:
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

    def split(self, x: np.ndarray, y: np.ndarray):
        folds = self.generate_folds(y_values=y)
        for train_indexes_folds, test_index_fold in self.splits:
            x_train = []
            y_train = []
            for train_index in train_indexes_folds:
                x_train.extend(x.take(folds[train_index], axis=0))
                y_train.extend(y.take(folds[train_index], axis=0))
            x_train = np.vstack(x_train)
            y_train = np.asarray(y_train)
            x_test = x.take(folds[test_index_fold], axis=0)
            y_test = y.take(folds[test_index_fold], axis=0)

            yield x_train, x_test, y_train, y_test
