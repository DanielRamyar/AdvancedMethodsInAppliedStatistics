import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])


def poly1(x, a):

    return a * x


def poly2(x, a, b):

    return a * x + b * x ** 2


def poly3(x, a, b, c):

    return a * x + b * x ** 2 + c * x ** 3


def poly4(x, a, b, c, d):

    return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4


a, b, c = curve_fit(poly3, x, y)[0]

# plt.plot(x, y, marker='o', linestyle='None')
# plt.plot(x, poly3(x, a, b, c))

# plt.show()


def my_chisquare(observed, expected, sigma):
    chi = np.sum((observed - expected) ** 2 / sigma ** 2)

    return chi


mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 100)
lnspc = np.linspace(min(s), max(s), len(s))

a, b, c = curve_fit(poly3, s, lnspc)[0]

binwidth = 0.05
n_bins = np.arange(min(s), max(s) + binwidth, binwidth)
plt.hist(s, bins=n_bins, density='True')

plt.plot(s, poly3(s, a, b, c))

plt.show()



print(my_chisquare(y, poly3(x, a, b, c), 0.5))
