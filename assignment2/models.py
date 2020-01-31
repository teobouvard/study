import numpy as np

from metrics import RMSE, log_RMSE

MIN_PER_NODE = 10
STOPPING_ERROR = 10

class Node:
    def __init__(self, error):
        self.size = None
        self.error = error
        self.parent = None

    def is_leaf(self):
        return (self.size < MIN_PER_NODE) or (self.error < STOPPING_ERROR) 


class Tree:
    def __init__(self):
        self.nodes = []
    
    def add_node(self, node):
        self.nodes.append(node)

    def has_converged(self):
        for n in self.nodes:
            if not n.is_leaf():
                return False
        return True
    
    def find_best_split(self):
        for n in self.nodes:

    def branch(self, x, y):
        if not self.has_converged():
            x = self.find_best_split(x, y)
            self.branch(x)
    
    def predict(self, x):
        pass


class DecisionTree:
    def __init__(self):
        self.model = Tree()

    def fit(self, x, y):
        if not self.model.nodes:
            err = RMSE(np.repeat(x.mean(), len(labels)), labels)
            n = Node(err)
            self.model.add_node(n)
        else:
            self.model.branch(x, y)

    def predict(self, x):
        return np.array([self.model.predict(_) for _ in x])


class RandomForest:
    def __init__(self, n_trees, **kwargs):
        self.trees = n_trees * [DecisionTree(**kwargs)]

    def fit(self, x, y):
        for t in self.trees:
            sample_features, sample_labels = self.sample(features, labels)
            t.fit(sample_features, sample_labels)

    def predict(self, x):
        predictions = np.array([t.predict(x) for t in self.trees])
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
    import pandas as pd
    np.random.seed(42)

    features = pd.read_csv('data/housing_price_train.csv', index_col=0)
    labels = features.pop('SalePrice')
    x_train, y_train, x_test, y_test = train_test_split(features, labels)

    model = DecisionTree()
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_test.values)
    error = RMSE(y_pred, y_test.values)
    log_error = log_RMSE(y_pred, y_test.values)


    print(error)
    print(log_error)
