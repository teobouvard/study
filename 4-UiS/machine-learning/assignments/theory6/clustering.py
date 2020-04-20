import numpy as np


def J(*clusters):
	error = 0
	for cluster in clusters:
		mean = cluster.mean(axis=1).reshape(2, -1)
		error += np.linalg.norm(cluster - mean) ** 2
	return error


x_1 = np.array([0, 0]).reshape(2, -1)
x_2 = np.array([1, 1]).reshape(2, -1)
x_3 = np.array([1, 0]).reshape(2, -1)
x_4 = np.array([2, 0.5]).reshape(2, -1)

X_1 = np.concatenate([x_1, x_2], axis=1)
X_2 = np.concatenate([x_3, x_4], axis=1)
err = J(X_1, X_2)
print('J for cluster 1 = ', err)

X_1 = np.concatenate([x_1, x_4], axis=1)
X_2 = np.concatenate([x_2, x_3], axis=1)
err = J(X_1, X_2)
print('J for cluster 2 = ', err)

X_1 = np.concatenate([x_1, x_2, x_3], axis=1)
X_2 = np.concatenate([x_4], axis=1)
err = J(X_1, X_2)
print('J for cluster 3 = ', err)

