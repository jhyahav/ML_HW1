import math

import numpy as np
import matplotlib.pyplot as plt

N = 200000
n = 20
matrix = np.random.randint(0, 2, (N, n))
row_means = matrix.sum(axis=1) / n


eps = np.linspace(0, 1, 50)
probs = [0] * len(eps)
bound = [0] * len(eps)
for i in range(len(eps)):
    exp = -2 * n * eps[i]**2
    mul = math.e ** exp
    bound[i] = 2 * mul
    cnt = 0
    for mean in row_means:
        cnt += abs(mean - 0.5) > eps[i]
    probs[i] = cnt / N
np_probs = np.array(probs)
np_bound = np.array(bound)

plt.plot(eps, np_probs, label="ℙ(|X̄ - 0.5| > ε)", color="black")
plt.plot(eps, np_bound, label="Hoeffding Bound", color="red")
plt.xlabel("ε")
plt.title("Empirical Probability vs. Hoeffding Bound")
plt.legend()
plt.show()
