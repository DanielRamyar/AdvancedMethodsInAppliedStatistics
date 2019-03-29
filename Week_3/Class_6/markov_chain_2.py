import numpy as np
import random
import matplotlib.pyplot as plt

start = 100
mu = 0.5 * start
sigma = 1

n = 100



new_x = np.array([])


for i in np.arange(n):
    temp = np.random.normal(mu, sigma, 1)
    mu = 0.5 * temp
    new_x = np.append(new_x, temp)


new_x_1 = np.array([])
start = -27
mu = 0.5 * start
sigma = 1

for i in np.arange(n):
    temp = np.random.normal(mu, sigma, 1)
    mu = 0.5 * temp
    new_x_1 = np.append(new_x_1, temp)

plt.plot(np.arange(n), new_x, 'ko', markersize=1, label='prior')
plt.plot(np.arange(n), new_x_1, 'ro', markersize=1, label='prior')
plt.show()

print(np.mean(new_x[10::]))