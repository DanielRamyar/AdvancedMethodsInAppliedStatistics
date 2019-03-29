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


posterior_flat = np.array([])
posterior_1overN = np.array([])

for i in np.arange(150, 2000):
    temp = my_hypergeo(k, big_k, n, i)
    posterior_flat = np.append(posterior_flat, temp)
    posterior_1overN = np.append(posterior_1overN, temp/i)

k = 15

posterior_flat_newk = np.array([])
posterior_1overN_newk = np.array([])

for i in np.arange(150, 2000):
    temp = my_hypergeo(k, big_k, n, i)
    posterior_flat_newk = np.append(posterior_flat_newk, temp)
    posterior_1overN_newk = np.append(posterior_1overN_newk, temp/i)

plt.plot(np.arange(150, 2000), posterior_flat/np.sum(posterior_flat), 'r')
plt.plot(np.arange(150, 2000), posterior_1overN/np.sum(posterior_1overN), 'r', linestyle='dashed')

plt.plot(np.arange(150, 2000), posterior_flat_newk/np.sum(posterior_flat_newk), 'b')
plt.plot(np.arange(150, 2000), posterior_1overN_newk/np.sum(posterior_1overN_newk), 'b', linestyle='dashed')
plt.show()
