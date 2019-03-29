import numpy as np


x = np.array([1.01, 1.3, 1.35, 1.44])


def my_gauss(x, mu, sigma):
    result = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return result


mu = 1.25
sigma = np.sqrt(0.11)


answer_1 = 1

for i in x:
    answer_1 = answer_1 * my_gauss(i, mu, sigma)

mu = 1.30
sigma = np.sqrt(0.5)


answer_2 = 1

for i in x:
    answer_2 = answer_2 * my_gauss(i, mu, sigma)

print(answer_1)
print(answer_2)


