import random
from typing import Generator, Set, Tuple, Any
import pickle

random.seed(10)


class Kfold:
    def __init__(self, k: int, shuffle=True):
        self.k = k
        self.shuffle = shuffle
        
    def generate_splits(self) -> Generator[tuple[tuple[int], int], None, None]:
        splits = []
        for index_fold_test in range(self.k):
            folds_for_training = []
            splits.append((folds_for_training, index_fold_test))
            for index_fold_train in range(self.k):
                if index_fold_train != index_fold_test:
                    folds_for_training.append(index_fold_train)
    
        splits = ((tuple(folds_for_training), fold_for_test) for folds_for_training, fold_for_test in splits)
        return splits
    
    def generate_folds(self, y_values: list[int]) -> set[tuple[int]]:
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

        return set([tuple(fold) for fold in folds])

#    def split(self, X, y):



if __name__ == "__main__":
    kfold = Kfold(3)
