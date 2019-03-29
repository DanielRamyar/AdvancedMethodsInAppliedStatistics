import numpy as np
import math
import matplotlib.pyplot as plt


k = 4
big_k = 50
n = 30


def my_binomial_coefficient(n, k):
    result = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return result

def my_hypergeo(k, K, n, N):
    result = my_binomial_coefficient(K, k) * my_binomial_coefficient(N - K, n - k) / my_binomial_coefficient(N, n)
    return result

def my_gauss(x, mu, sigma):
    result = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return result


posterior_flat = np.array([])
posterior = np.array([])
prior = np.array([])
mu = 500
sigma = 500 * np.sqrt((300/5000) ** 2 + (1/10) ** 2)
my_start = big_k + n - k


for i in np.arange(my_start, 2000):
    temp = my_hypergeo(k, big_k, n, i)
    
    posterior = np.append(posterior, temp * my_gauss(i, mu, sigma))
    posterior_flat = np.append(posterior_flat, temp)

    prior = np.append(prior, my_gauss(i, mu, sigma))

k=8
posterior_flat_newk = np.array([])
posterior_newk = np.array([])
for i in np.arange(my_start, 2000):
    temp = my_hypergeo(k, big_k, n, i)
    
    posterior_newk = np.append(posterior_newk, temp * my_gauss(i, mu, sigma))
    posterior_flat_newk = np.append(posterior_flat_newk, temp)



plt.plot(np.arange(my_start, 2000), posterior_flat/np.sum(posterior_flat), 'r', label='flat/LLH k=4')
plt.plot(np.arange(my_start, 2000), posterior/np.sum(posterior), 'r', linestyle='dashed', label='gaussian k=4')

plt.plot(np.arange(my_start, 2000), posterior_flat_newk/np.sum(posterior_flat_newk), 'b', label='flat/LLH k=8')
plt.plot(np.arange(my_start, 2000), posterior_newk/np.sum(posterior_newk), 'b', linestyle='dashed', label='gaussian k=8')
plt.plot(np.arange(my_start, 2000), prior/np.sum(prior), 'k', label='prior')
plt.legend()
plt.show()
