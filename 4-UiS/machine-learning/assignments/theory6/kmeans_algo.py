import numpy as np

def initial_assignment(points, means):
	c_1 = []
	c_2 = []
	for i, p in enumerate(points):
		if np.linalg.norm(p-means[0])**2 < np.linalg.norm(p-m_2)**2:
			c_1.append(i+1)
		else:
			c_2.append(i+1)
	return c_1, c_2


x_1 = np.array([1.0, 1.0]).reshape(2, 1)
x_2 = np.array([3.0, 1.0]).reshape(2, 1)
x_3 = np.array([2.0, 3.0]).reshape(2, 1)
x_4 = np.array([1.0, 4.5]).reshape(2, 1)
x_5 = np.array([2.5, 1.5]).reshape(2, 1)
x_6 = np.array([3.0, 3.0]).reshape(2, 1)
x_7 = np.array([3.0, 4.0]).reshape(2, 1)

m_1 = np.array([2.0, 3.0]).reshape(2, 1)
m_2 = np.array([3.0, 1.0]).reshape(2, 1)

points = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
means = [m_1, m_2]

clusters = initial_assignment(points, means)
print(f'{clusters=}')
mask_0 = np.array(clusters[0]) - 1
mask_1 = np.array(clusters[1]) - 1
means[0] = np.mean(np.concatenate(points)[mask_0], axis=0)
means[1] = np.mean(np.concatenate(points)[mask_1], axis=0)

rng = [1, 2, 3, 7, 6, 4, 5, 4, 7, 3, 2, 5, 6, 1]

for idx in rng:
	chosen_sample = points[idx-1]
	distances = [np.linalg.norm(m-chosen_sample)**2 for m in means]
	i = np.argmin(distances)
	if len(clusters[i]) != 1:
		j = (i+1)%2
		new_mean_i = means[i] - (chosen_sample-means[i]) / (len(clusters[i]) - 1)
		new_mean_j = means[j] + (chosen_sample-means[j]) / (len(clusters[j]) + 1)
		decrease = (len(clusters[i])/(len(clusters[i])-1)) * np.linalg.norm(chosen_sample-new_mean_i)**2
		increase = (len(clusters[j])/(len(clusters[j])+1)) * np.linalg.norm(chosen_sample-new_mean_j)**2
		if decrease > increase:
			f = 0 if idx in clusters[0] else 1
			t = (f+1)%2
			print(f'Transfer x_{idx} from {i+1} to {j+1}')
			clusters[i].remove(idx)
			clusters[j].append(idx)
			means[i] = new_mean_i
			means[j] = new_mean_j
		