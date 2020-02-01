import numpy as np

from metrics import RMSE, log_RMSE

MIN_PER_NODE = 10

class Node:
    def __init__(self, x, y, depth=0, max_depth=10, benchmark=False):
        self.max_depth = max_depth
        self.depth = depth
        self.size = len(y)
        self.is_leaf = (self.size < MIN_PER_NODE) or (self.depth > max_depth)
        self.value = y.mean()
        self.error = RMSE(np.repeat(self.value, self.size), y)
        if not self.is_leaf and not benchmark:
            self.split_idx, self.split_value, self.split_type = self.find_best_split(x, y)
            self.left, self.right = self.grow(x, y)
    
    def find_best_split(self, x, y):
        split_idx = None
        split_value = None
        split_type = None
        best_error_reduction = 0

        for idx in range(x.shape[1]):
            # check that all feature values for this index are of the same type
            if len(set(type(_) for _ in x[:, idx])) > 1:
                continue

            possible_values = sorted(set(x[:, idx]))

            if isinstance(x[0, idx], (int, float)):
                for i in range(len(possible_values) - 1):
                    threshold = np.mean([possible_values[i], possible_values[i+1]])
                    left_split = Node(x, y[x[:, idx] <= threshold], benchmark=True)
                    right_split = Node(x, y[x[:, idx] > threshold], benchmark=True)
                    error_reduction = self.error - (left_split.size*left_split.error+right_split.size*right_split.error)/(2*self.size)
                    if error_reduction > best_error_reduction:
                        best_error_reduction = error_reduction
                        split_idx = idx
                        split_value = threshold
                        split_type = 'numeric'

            elif isinstance(x[0, idx], str):
                for value in possible_values:
                    left_split = Node(x, y[x[:, idx] == value], benchmark=True)
                    right_split = Node(x, y[x[:, idx] != value], benchmark=True)
                    error_reduction = self.error - (left_split.size*left_split.error+right_split.size*right_split.error)/2
                    if error_reduction > best_error_reduction:
                        best_error_reduction = error_reduction
                        split_idx = idx
                        split_value = value
                        split_type = 'categorical'
            else:
                print('AH')
                split_type = 'wtf'
        
        return split_idx, split_value, split_type
    
    def grow(self, x, y):
        if self.split_type == 'numeric':
            left_mask = x[:, self.split_idx] <= self.split_value
        elif self.split_type == 'categorical':
            left_mask = x[:, self.split_idx] == self.split_value
        else:
            print('ah')

        left_node = Node(x[left_mask], y[left_mask], depth=self.depth+1, max_depth=self.max_depth)
        right_node = Node(x[np.invert(left_mask)], y[np.invert(left_mask)], depth=self.depth+1, max_depth=self.max_depth)
        return left_node, right_node

class DecisionTree:
    def __init__(self, max_depth=10):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, x, y):
        self.tree = Node(x, y, max_depth=self.max_depth)

    def predict(self, x):
        return np.array([self.predict_instance(_) for _ in x])

    def predict_instance(self, x):
        node = self.tree
        while not node.is_leaf:
            if node.split_type == 'numeric':
                if x[node.split_idx] < node.split_value:
                    node = node.left
                else:
                    node = node.right
            elif node.split_type == 'categorical':
                if x[node.split_idx] == node.split_value:
                    node = node.left
                else:
                    node = node.right
        return node.value


class RandomForest:
    def __init__(self, n_trees, **kwargs):
        self.trees = n_trees * [DecisionTree(**kwargs)]

    def fit(self, x, y):
        for t in self.trees:
            sample_features, sample_labels = self.sample(features, labels)
            t.fit(sample_features, sample_labels)

    def predict(self, x):
        predictions = np.array([t.predict(x) for t in self.trees])
        return predictions.mean() # maybe different axis

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

    model = DecisionTree(max_depth=5)
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_test.values)
    baseline_error = RMSE(np.repeat(y_train.values.mean(), len(y_test)), y_test.values)
    error = RMSE(y_pred, y_test.values)
    log_error = log_RMSE(y_pred, y_test.values)


    print(f'RMSE : {error} - {error/baseline_error:%} of baseline error')
    #print(log_error)
