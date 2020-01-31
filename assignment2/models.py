import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.model = []

    def fit(self, features, labels):
        self.mean_value = labels.mean()

    def predict(self, features):
        return np.repeat(self.mean_value, len(features))

class RandomForest:
    def __init__(self, n_trees, **kwargs):
        self.trees = n_trees * [DecisionTreeClassifier(**kwargs)]

    def fit(self, features, labels):
        for t in self.trees:
            sample_features, sample_labels = self.sample(features, labels)
            t.fit(sample_features, sample_labels)

    def predict(self, features):
        predictions = np.array([t.predict(features) for t in self.trees])
        return predictions.mean(axis=1)

    def sample(self, x, y, frac=0.8):
        # sanity check
        assert len(x) == len(y)
        n_samples = int(frac * len(x))
        all_indices = np.arange(len(x))
        indices = np.random.choice(all_indices, n_samples, replace=False)
        return x[indices], y[indices]


if __name__ == '__main__':
    from utils import train_test_split
    from metrics import RMSE, log_RMSE
    import pandas as pd
    np.random.seed(42)

    features = pd.read_csv('data/housing_price_train.csv', index_col=0)
    labels = features.pop('SalePrice')
    x_train, y_train, x_test, y_test = train_test_split(features, labels)

    model = DecisionTreeClassifier()
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_test.values)
    error = RMSE(y_pred, y_test.values)
    log_error = log_RMSE(y_pred, y_test.values)


    print(error)
    print(log_error)
