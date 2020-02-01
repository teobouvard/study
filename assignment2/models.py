import numpy as np
#np.seterr(all='raise') 

from multiprocessing import Pool
from metrics import RMSE, log_RMSE

class Node:

    def __init__(self, x, y, depth=0, max_depth=10, min_node_size=10, benchmark=False):
        self.max_depth = max_depth
        self.depth = depth
        self.size = len(y)
        self.is_leaf = (self.size < min_node_size) or (self.depth > max_depth)
        self.value = y.mean() if self.size > 0 else 0
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

            # numeriacal attributes
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

            # categorical attributes
            elif isinstance(x[0, idx], str):
                for value in possible_values:
                    left_split = Node(x, y[x[:, idx] == value], benchmark=True)
                    right_split = Node(x, y[x[:, idx] != value], benchmark=True)
                    error_reduction = self.error - (left_split.size*left_split.error+right_split.size*right_split.error)/(2*self.size)
                    if error_reduction > best_error_reduction:
                        best_error_reduction = error_reduction
                        split_idx = idx
                        split_value = value
                        split_type = 'categorical'

            else:
                raise ValueError(f'Attribute {x[0, idx]} is of type {type(x[0, idx])} but should be int, float or str')
        
        return split_idx, split_value, split_type
    
    def grow(self, x, y):
        if self.split_type == 'numeric':
            left_mask = x[:, self.split_idx] <= self.split_value
        elif self.split_type == 'categorical':
            left_mask = x[:, self.split_idx] == self.split_value
        else:
            raise ValueError(f'Split type is {self.split_type} but sould be numeric or categorical')

        left_node = Node(x[left_mask], y[left_mask], depth=self.depth+1, max_depth=self.max_depth)
        right_node = Node(x[np.invert(left_mask)], y[np.invert(left_mask)], depth=self.depth+1, max_depth=self.max_depth)
        return left_node, right_node

class DecisionTree:
    def __init__(self, max_depth=10, min_node_size=10):
        self.tree = None
        self.max_depth = max_depth
        self.min_node_size = min_node_size

    def fit(self, x, y):
        self.tree = Node(x, y, max_depth=self.max_depth, min_node_size=self.min_node_size)
        return self # so that multiprocess training can collect the results

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
    def __init__(self, n_trees=100, subsample_size=0.6, **kwargs):
        self.trees = [DecisionTree(**kwargs) for _ in range(n_trees)]
        self.subsample_size = subsample_size

    def fit(self, x, y):
        subsamples = [self.sample(x, y, self.subsample_size) for _ in self.trees]
        xs = [_[0] for _ in subsamples]
        ys = [_[1] for _ in subsamples]
        #with Pool() as p:
        #    self.trees = p.starmap(DecisionTree.fit, zip(self.trees, xs, ys))
        for t in self.trees:
            t.fit(*self.sample(x, y, self.subsample_size))

    def predict(self, x):
        predictions = np.array([t.predict(x) for t in self.trees])
        return predictions.mean(axis=0)

    def sample(self, x, y, frac):
        assert len(x) == len(y)
        n_samples = int(frac * len(x))
        all_indices = np.arange(len(x))
        indices = np.random.choice(all_indices, n_samples, replace=False)
        return x[indices], y[indices]


if __name__ == '__main__':
    from utils import train_test_split
    import pandas as pd
    np.random.seed(42)

    train_data = pd.read_csv('data/housing_price_train.csv', index_col=0)
    test_data = pd.read_csv('data/housing_price_test.csv', index_col=0)

    attrs = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']
    train_data[attrs] = train_data[attrs].fillna('None')
    test_data[attrs] = test_data[attrs].fillna('None')

    attr = 'LotFrontage'
    train_data[attr] = train_data[attr].fillna(train_data[attr].mean())
    test_data[attr] = test_data[attr].fillna(test_data[attr].mean())

    attr = 'GarageYrBlt'
    train_data.drop(attr, axis='columns', inplace=True)
    test_data.drop(attr, axis='columns', inplace=True)

    attrs = ['MasVnrArea', 'Electrical']
    train_data.dropna(axis='index', subset=attrs, inplace=True)
    test_data.dropna(axis='index', subset=attrs, inplace=True)

    train_labels = train_data.pop('SalePrice')
    x_train, y_train, x_val, y_val = train_test_split(train_data, train_labels)

    model = RandomForest(n_trees=10, max_depth=5)
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_val.values)

    baseline_error = RMSE(np.repeat(y_train.values.mean(), len(y_val)), y_val.values)
    error = RMSE(y_pred, y_val.values)
    log_error = log_RMSE(y_pred, y_val.values)
    print(f'RMSE : {error} - l-RMSE : {log_error} - {error/baseline_error:%} of baseline error')