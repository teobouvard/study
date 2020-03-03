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
        n_points = np.size(self.points[label])
        hn = self.h / np.sqrt(n_points)
        dim = self.points[label].shape[-1]
        vn = pow(hn, dim)
        centered = ((p - x) / hn for p in self.points[label])
        return sum(self._phi(u, dim) for u in centered) / (n_points * vn)

    def _phi(self, u, dim):
        return pow(2 * np.pi, dim) * np.exp(-0.5 * u.T @ u)


class SKLParzenClassifier:
    def __init__(self, h):
        self.classes = []
        self.h = h
        self.distributions = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            self.classes.append(label)
            class_idx = np.where(y == label)
            hn = self.h / np.sqrt(len(class_idx))
            dist = KernelDensity(bandwidth=hn)
            self.distributions[label] = dist.fit(x[class_idx], y[class_idx])

    def predict(self, x):
        return np.array([self._predict_instance(_.reshape(1, -1)) for _ in x])

    def _predict_instance(self, x):
        return max(self.classes, key=lambda label: self.distributions[label].score(x))


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


if __name__ == "__main__":
    # x, y = datasets.load_digits(return_X_y=True)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    from utils import normalize_data

    _, _, _, _, X_3D3cl_ms, _, _, _, _, Y_3D3cl_ms = np.load(
        "data/lab4_2.p", allow_pickle=True
    )
    x_train, x_test, y_train, y_test = normalize_data(X_3D3cl_ms, Y_3D3cl_ms)

    # model = MLClassifier()
    # model = SKLParzenClassifier(h=1)
    # model = KNNClassifier(k=5)
    # model = ParzenClassifer(h=5)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # cm = confusion_matrix(y_pred, y_test, normalize=True)
    # acc = accuracy_score(y_pred, y_test)
    # print(f"Accuracy : {acc:.2f}")
    # print(f"Confusion matrix : \n{cm}")

    import pandas as pd
    from utils import error_score, confusion_matrix

    def train_test_evaluate(models):
        global x_train, x_test, y_train, y_test

        results = {
            "P(error)": {},
        }

        for desc, model in models.items():
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            err_train = error_score(y_pred_train, y_train)
            err_test = error_score(y_pred_test, y_test)
            cm_train = confusion_matrix(y_pred_train, y_train)
            cm_test = confusion_matrix(y_pred_test, y_test)

            variant = ["Reclassification", "Testing"]
            error = [err_train, err_test]
            confusion = [cm_train, cm_test]

            for var, err, cm in zip(variant, error, confusion):
                results["P(error)"][f"{desc} {var}"] = f"{err:.2f}"
                for i, p_correct in enumerate(cm.diagonal()):
                    true_positives = f"P( correct | w_{i+1})"
                    if true_positives not in results:
                        results[true_positives] = {}
                    results[f"P( correct | w_{i+1})"][
                        f"{desc} {var}"
                    ] = f"{p_correct:.2f}"

        return pd.DataFrame(results)

    models = {
        "Maximum likelihood": MLClassifier(),
        "Parzen h1 = 0.1": ParzenClassifer(h=0.1),
        "Parzen h1 = 5.0": ParzenClassifer(h=5.0),
        "Nearest neighbours k = 1": KNNClassifier(k=1),
        "Nearest neighbours k = 5": KNNClassifier(k=5),
    }
    train_test_evaluate(models)
