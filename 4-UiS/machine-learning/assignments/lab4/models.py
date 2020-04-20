from collections import Counter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
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
        vec = x - self.mu[label]
        inv = np.linalg.inv(self.cov[label])
        det = np.linalg.det(self.cov[label])
        prior = self.prior[label]
        dim = x.shape[-1]
        return (
            -0.5 * (vec.T @ inv @ vec)
            - 0.5 * dim * np.log(2 * np.pi)
            - 0.5 * np.log(det)
            + np.log(prior)
        )


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
        n_points = len(self.points[label])
        hn = self.h / np.sqrt(n_points)
        dim = self.points[label].shape[-1]

        vn = pow(hn, dim)
        t = vn * pow(2 * np.pi, dim / 2)
        centered = ((x - p) / hn for p in self.points[label])
        kn = sum((1 / t) * self._phi(p) for p in centered)

        return kn / n_points

    def _phi(self, u):
        return np.exp(-(u.T @ u) / 2)


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
        # note that this nested loops are highly inefficient, but I find them more
        # self-explanatory than doing vectorized distances computations.
        # ideally, points should be stored in an adequate data structure, such as
        # a k-d tree, for more efficient neighbour queries
        distances = []
        for c in self.classes:
            for p in self.points[c]:
                distances.append((np.linalg.norm(p - x), c))
        neighbours = sorted(distances, key=lambda x: x[0])
        neighbours = Counter(_[1] for _ in neighbours[: self.k])
        return neighbours.most_common(1)[0][0]
