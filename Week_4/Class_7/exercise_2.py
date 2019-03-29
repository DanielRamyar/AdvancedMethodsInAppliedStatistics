import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare


fname = 'Lecture8_LLH_Ratio_2_data.txt'

loaded_file = np.loadtxt(fname)

x_1 = loaded_file[:, 0]

fname = 'Lecture8_LLH_Ratio_2a_data.txt'

loaded_file = np.loadtxt(fname)

x_2 = loaded_file[:, 0]


def norm(x, alpha, beta):
    upper_lim = (max(x) + 1 / 2 * alpha * (max(x) ** 2) +
                 1 / 3 * beta * (max(x) ** 3))
    lower_lim = (min(x) + 1 / 2 * alpha * (min(x) ** 2) +
                 1 / 3 * beta * (min(x) ** 3))

    integral = upper_lim - lower_lim
    return integral


def norm_alternate(x, alpha, beta, gamma):
    upper_lim = (max(x) +
                 1 / 2 * alpha * (max(x) ** 2) +
                 1 / 3 * beta * (max(x) ** 3) -
                 1 / 6 * gamma * (max(x) ** 6))
    lower_lim = (min(x) +
                 1 / 2 * alpha * (min(x) ** 2) +
                 1 / 3 * beta * (min(x) ** 3) -
                 1 / 6 * gamma * (min(x) ** 6))

    integral = upper_lim - lower_lim
    return integral


def my_func(x, alpha, beta):
    return (1 + alpha * x + beta * x ** 2) / norm(x, alpha, beta)


def my_func_alternate(x, alpha, beta, gamma):
    return ((1 + alpha * x + beta * x ** 2 - gamma * x ** 5) /
            norm_alternate(x, alpha, beta, gamma))


binwidth = 0.05
n_bins = np.arange(min(x_1), max(x_1) + binwidth, binwidth)
y, x, _ = plt.hist(x_1, bins=n_bins, normed=1)

x = x + binwidth / 2

popt, pcov = curve_fit(my_func, x[0:-1], y)

plt.plot(x, my_func(x, *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.plot(x[0:-1], y, linestyle='none', marker='.')


print(chisquare(y, my_func(x[0:-1], *popt), ddof=len(y)-2))

LLH = np.sum(np.log(my_func(x_1, *popt)))

print(LLH)

#####

popt_alternate, pcov = curve_fit(my_func_alternate, x[0:-1], y)

plt.plot(x, my_func_alternate(x, *popt_alternate), 'b--',
         label='fit: a=%5.3f, b=%5.3f, g=%5.3f' % tuple(popt_alternate))

plt.plot(x[0:-1], y, linestyle='none', marker='.')
plt.legend()
plt.show()


print(chisquare(y, my_func_alternate(x[0:-1], *popt_alternate), ddof=len(y)-2))

LLH1 = np.sum(np.log(my_func_alternate(x_1, *popt_alternate)))

print(LLH1)

LLH_ratio = -2 * (LLH - LLH1)
print('Likelyhood Ratio', LLH_ratio)
