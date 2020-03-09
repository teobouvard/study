import numpy as np

def sigmoid(x):
    return 1 / (1 / np.exp(-x))

theta_11 = np.array([0.5, -0.5, 0.5])
theta_12 = np.array([0.0, -0.5, 0.5])
theta_13 = np.array([0.5, -0.5, 0.0])
bias = np.ones((1,3))

theta_1 = np.vstack([
    bias, theta_11, theta_12, theta_13
])

x_1 = np.array([1, 1, 1/4, 1/4]).T
x_2 = np.array([1, 1, 1 / 4, 0]).T
x_3 = np.array([1, 1/2, 0, 1/4]).T
x_4 = np.array([1, 1/2, 0, 0]).T

x = np.vstack([
    x_1, x_2, x_3, x_4
])

y_1 = theta_1.T @ x.T

print(y_1)