import numpy as np

def train_test_split(x, y, test_ratio=0.2):
    # sanity check
    assert len(x) == len(y)
    n_samples = int(test_ratio * len(x))
    all_indices = np.arange(len(x))
    test_indices = np.random.choice(all_indices, n_samples, replace=False)
    train_indices = all_indices[np.isin(all_indices, test_indices, invert=True)]
    return x.iloc[train_indices], y.iloc[train_indices], x.iloc[test_indices], y.iloc[test_indices]
