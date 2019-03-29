import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare


fname = 'MLE_Variance_data.txt'

loaded_file = np.loadtxt(fname)

x_1 = loaded_file[:, 0]
x_2 = loaded_file[:, 1]


def norm(x, alpha, beta):
    upper_lim = (max(x) + 1 / 2 * alpha * (max(x) ** 2) +
                 1 / 3 * beta * (max(x) ** 3))
    lower_lim = (min(x) + 1 / 2 * alpha * (min(x) ** 2) +
                 1 / 3 * beta * (min(x) ** 3))

    integral = upper_lim - lower_lim
    return integral


def my_func(x, alpha, beta):
    return (1 + alpha * x + beta * x ** 2) / norm(x, alpha, beta)


binwidth = 0.05
n_bins = np.arange(min(x_1), max(x_1) + binwidth, binwidth)
y, x, _ = plt.hist(x_1, bins=n_bins, normed=1)

x = x + binwidth / 2

popt, pcov = curve_fit(my_func, x[0:-1], y)

plt.plot(x, my_func(x, *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.plot(x[0:-1], y, linestyle='none', marker='.')
plt.legend()
plt.show()


print(chisquare(y, my_func(x[0:-1], *popt), ddof=len(y)-2))
