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

print(f'{cluster_1=}')
print(f'{cluster_2=}')

m_1 = np.array([9/4, 29/8])
m_2 = np.array([13/6, 7/6])


distances = np.zeros((len(points), len(points)))
for i, p1 in enumerate(points):
	for j, p2 in enumerate(points):
		distances[i, j] = np.linalg.norm(p1 - p2)**2
print(distances)

chosen = x_1 

increase = len(cluster_1)/(len(cluster_1)+1) * np.linalg.norm(chosen - m_1)**2
decrease = len(cluster_2)/(len(cluster_2)-1) * np.linalg.norm(chosen - m_2)**2

print(decrease)
print(increase)

if decrease > increase:
	print('Transfer x_1')
	cluster_2.remove(1)
	cluster_1.append(1)
	

