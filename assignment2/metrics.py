import numpy as np

def RMSE(x, y):
    return np.sqrt(((x - y) ** 2).mean())

def log_RMSE(x, y):
    return np.sqrt(((np.log(x) - np.log(y)) ** 2).mean())


if __name__ == '__main__':
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    test_error = RMSE(y_pred, y_true)
    np.testing.assert_almost_equal(test_error, 0.61237, decimal=4)