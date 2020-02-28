from collections import Counter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import accuracy_score, confusion_matrix


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
    def __init__(self, h):
        self.classes = []
        self.h = h
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
        n_points = np.size(self.points[label])
        hn = self.h / np.sqrt(n_points)
        centered = ((p - x) for p in self.points[label])
        return sum(self._phi(u, hn) for u in centered) / n_points

    def _phi(self, u, hn):
        # we don't need to take into account the constants
        # which are identical for all classes
        return hn ** 2 * np.exp(-(0.5 / hn) * u.T @ u)


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
        neighbours = sorted(distances, key=lambda x: x[0])
        neighbours = Counter(_[1] for _ in neighbours[: self.k])
        return neighbours.most_common(1)[0][0]


if __name__ == "__main__":
    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    model = MLClassifier()
    # model = KNNClassifier(k=5)
    # model = ParzenClassifer(h=100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_pred, y_test, normalize=True)
    acc = accuracy_score(y_pred, y_test)
    print(f"Accuracy : {acc:.2f}")
    print(f"Confusion matrix : \n{cm}")
