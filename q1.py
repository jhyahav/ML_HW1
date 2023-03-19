import numpy as np

N = 200000
n = 20
matrix = np.random.randint(0, 2, (N, n))
row_means = matrix.sum(axis=1) / n


eps = np.linspace(0, 1, 50)
probs = [0] * len(eps)
for i in range(len(eps)):
    cnt = 0
    for mean in row_means:
        cnt += abs(mean - 0.5) > eps[i]
    probs[i] = cnt / N
np_probs = np.array(probs)