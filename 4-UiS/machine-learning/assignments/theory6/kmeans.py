import numpy as np

x_1 = np.array([1.0, 1.0])
x_2 = np.array([3.0, 1.0])
x_3 = np.array([2.0, 3.0])
x_4 = np.array([1.0, 4.5])
x_5 = np.array([2.5, 1.5])
x_6 = np.array([3.0, 3.0])
x_7 = np.array([3.0, 4.0])

m_1 = np.array([2.0, 3.0])
m_2 = np.array([3.0, 1.0])

points = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
cluster_1 = []
cluster_2 = []

for i, p in enumerate(points):
	if np.linalg.norm(p-m_1) < np.linalg.norm(p-m_2):
		cluster_1.append(i+1)
	else:
		cluster_2.append(i+1)

print('bork')

rng = [1, 2, 3, 7, 6, 4, 5]