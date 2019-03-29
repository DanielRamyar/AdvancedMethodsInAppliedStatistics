import numpy as np
import math
import matplotlib.pyplot as plt


k = 10
big_k = 100
n = 60


def my_binomial_coefficient(n, k):
    result = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return result

def my_hypergeo(k, K, n, N):
    result = my_binomial_coefficient(K, k) * my_binomial_coefficient(N - K, n - k) / my_binomial_coefficient(N, n)
    return result


posterior = np.array([])

for i in np.arange(150, 2000):
    temp = my_hypergeo(k, big_k, n, i)
    posterior = np.append(posterior, temp)

plt.plot(np.arange(150, 2000), posterior)
plt.show()
