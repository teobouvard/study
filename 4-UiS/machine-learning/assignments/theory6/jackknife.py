import numpy as np


seq = np.random.randint(0, 100, 1000)
print(f'Mean of sequence = {seq.mean()}')

subseq_means = np.zeros(seq.shape)
index = np.arange(len(seq))
for i in index:
    jk_sample = seq[index != i]
    subseq_means[i] = jk_sample.mean()

print(f"Jackknife estimate of the mean = {subseq_means.mean()}")