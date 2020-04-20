from sklearn.neighbors import KernelDensity
import numpy as np

class ParzenClassifier:
    def __init__(self, h, kernel):
        self.classes = []
        self.h = h
        self.distributions = {}
        self.priors = {}

        if kernel not in ['gaussian','tophat','epanechnikov','exponential','linear','cosine']:
            raise NotImplementedError('Not a valid kernel')
        self.kernel = kernel

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            self.classes.append(label)
            class_idx = (y == label)
            self.priors[label] = class_idx.sum()/len(y)
            hn = self.h / np.sqrt(len(class_idx))
            dist = KernelDensity(bandwidth=hn, kernel=self.kernel)
            self.distributions[label] = dist.fit(x[class_idx], y[class_idx])

    def predict(self, x):
        return np.array([self._predict_instance(_.reshape(1, -1)) for _ in x])

    def _predict_instance(self, x):
        return max(self.classes, key=lambda label: self.priors[label] * np.exp(self.distributions[label].score(x)))