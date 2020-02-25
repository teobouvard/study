from collections import Counter

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


class MLClassifier:
    def __init__(self):
        self.classes = []
        self.prior = {}
        self.mu = {}
        self.cov = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.prior[label] = np.size(class_idx) / np.size(y)
            self.mu[label] = np.mean(x[class_idx], axis=0)
            self.cov[label] = np.cov(x[class_idx], rowvar=False)

    def predict(self, x):
        return np.array([self._predict_instance(_) for _ in x])

    def _predict_instance(self, x):
        return max(self.classes, key=lambda label: self._discriminant(x, label))

    def _discriminant(self, x, label):
        a = x - self.mu[label]
        b = np.linalg.inv(self.cov[label])
        c = np.linalg.det(self.cov[label])
        d = self.prior[label]
        return -0.5 * (a.T @ b @ a) - 0.5 * np.log(c) + np.log(d)


class ParzenClassifer:
    def __init__(self, h, kernel="cube"):
        self.classes = []
        self.kernel = kernel
        self.h = h
        self.n_training = {}
        self.points = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.points[label] = x[class_idx]

    def predict(self, x):
        return np.array([self._predict_instance(_) for _ in x])

    def _predict_instance(self, x):
        return max(self.classes, key=lambda label: self._discriminant(x, label))

    def _discriminant(self, x, label):
        # TODO ask for parzen window function clarifications
        n_points = np.size(self.points[label])
        hn = self.h / np.sqrt(n_points)
        centered = ((p - x) / hn for p in self.points[label])
        return sum(self._phi(u) / hn for u in centered) / n_points

    def _phi(self, u):
        if self.kernel == "cube":
            return self._hypercube(u)
        elif self.kernel == "gaussian":
            return self._gaussian(u)
        else:
            raise NotImplementedError(f"{self.kernel} kernel not implemented")

    def _hypercube(self, u):
        if max(np.abs(u)) <= 0.5:
            return 1
        return 0

    def _gaussian(self, u):
        if max(np.abs(u)) <= 0.5:
            return 1
        return 0


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.classes = []
        self.points = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.points[label] = x[class_idx]

    def predict(self, x):
        return np.array([self._predict_instance(_) for _ in x])

    def _predict_instance(self, x):
        distances = []
        for c in self.classes:
            for p in self.points[c]:
                distances.append((np.linalg.norm(p - x), c))
        neighbours = sorted(distances, key=lambda x: x[0])[: self.k]
        neighbours = Counter(_[1] for _ in neighbours)
        return neighbours.most_common(1)[0][0]


if __name__ == "__main__":
    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    # model = MLClassifier()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # cm = confusion_matrix(y_pred, y_test)
    # acc = accuracy_score(y_pred, y_test)
    # print(f"Accuracy : {acc}")
    # print(f"Confusion matrix : {cm}")

    model = ParzenClassifer(h=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    print(f"Accuracy : {acc:.2f}")
    print(f"Confusion matrix : \n{cm}")

    # model = KNNClassifier(k=3)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # cm = confusion_matrix(y_pred, y_test)
    # acc = accuracy_score(y_pred, y_test)
    # print(f"Accuracy : {acc}")
    # print(f"Confusion matrix : \n{cm}")
