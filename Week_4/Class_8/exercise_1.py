import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def my_gauss(x, mu, sigma):
    result = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return result


def my_kernel(data, x, h):

    kernels = np.array([])

    for k in x:
        temp = 0
        x_diff = 0

        for i in data:
            x_diff = (k - i) / h
            if abs(x_diff) < 1:
                temp += 1 / (2 * h)
            else:
                pass

        temp = temp / len(data)
        kernels = np.append(kernels, temp)

    return kernels


def my_kernel_gauss(data, x, h):

    kernels = np.array([])
    mu = 0
    sigma = 3

    for k in x:
        temp = 0
        x_diff = 0

        for i in data:
            x_diff = (k - i)
            temp += my_gauss(x_diff, mu, sigma)

        temp = temp / len(data)
        kernels = np.append(kernels, temp)

    return kernels


data = np.array([1, 2, 5, 6, 12, 15, 16, 16, 22, 22, 22, 23])

h = 1.5

x = np.linspace(-10, 35, 1000)

p = my_kernel(data, x, h)
p2 = my_kernel_gauss(data, x, h)
kde = KernelDensity(kernel='gaussian', bandwidth=2 * h).fit(data[:, None])
p3 = np.exp(kde.score_samples(x[:, None]))

plt.plot(x, p, label='Box')
plt.plot(x, p2, label='Gauss_homemade')
plt.plot(x, p3, label='KDE scipy')
plt.legend()
plt.show()
