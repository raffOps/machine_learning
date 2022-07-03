import numpy as np

from machine_learning import distance


class KNeighborsClassifier:

    def __init__(self, k: int, distance_function=distance.get_euclidean_distance):
        self.y_train = None
        self.X_train = None
        self.k = k
        self.distance_function = distance_function
        self.distance_matrix = None

    def fit(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def __get_k_nearest_neighbors(self, distance_matrix) -> list[list[int]]:
        nearest_neighbors = []
        for x_test in distance_matrix.columns:
            nearest_neighbors_from_x_test = distance_matrix[x_test].sort_values(ascending=True).head(
                self.k).index.tolist()
            nearest_neighbors.append(nearest_neighbors_from_x_test)

        return nearest_neighbors

    def predict(self, X: np.ndarray) -> np.ndarray:
        distance_matrix = distance.get_distance_matrix(self.X_train, X, distance_function=self.distance_function)
        nearest_neighbors = self.__get_k_nearest_neighbors(distance_matrix)

        predicted_classes = []
        for x in range(X.shape[0]):
            predict_class = np.bincount(self.y_train[nearest_neighbors[x]]).argmax()
            predicted_classes.append(predict_class)

        return np.array(predicted_classes)
