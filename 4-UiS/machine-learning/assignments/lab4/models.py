import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


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
        centered_samples = ((p - x) / self.h for p in self.points[label])
        in_window = sum(self._phi(u) for u in centered_samples)
        return in_window / np.size(self.points[label])

    def _phi(self, u):
        if self.kernel == "cube":
            return self._hypercube(u)
        elif self.kernel == "gaussian":
            # TODO ?
            raise NotImplementedError("Gaussian kernel not yet implemented")
        else:
            raise NotImplementedError(f"{self.kernel} kernel not yet implemented")

    def _hypercube(self, u):
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
        distances = {}
        neighbours = {}
        for c in self.classes:
            neighbours[c] = 0
            distances[c] = sorted(np.linalg.norm(p - x) for p in self.points[c])
        for i in range(self.k):
            closest = min(distances.items(), key=lambda x: x[1][0])[0]
            neighbours[closest] += 1
        return max(neighbours.items(), key=lambda x: x[1])[0]


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

    # model = ParzenClassifer(h=5)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # cm = confusion_matrix(y_pred, y_test)
    # acc = accuracy_score(y_pred, y_test)
    # print(f"Accuracy : {acc}")
    # print(f"Confusion matrix : \n{cm}")

    model = KNNClassifier(k=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    print(f"Accuracy : {acc}")
    print(f"Confusion matrix : \n{cm}")
