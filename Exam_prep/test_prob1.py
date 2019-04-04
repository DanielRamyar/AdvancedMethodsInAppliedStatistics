import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare, chi2, binom, poisson


def f_1(x, a):
    return (1 / (x + 5)) * np.sin(a * x)


def f_2(x, a):
    return np.sin(a * x) + 1


def f_3(x, a):
    return np.sin(a * (x ** 2))


def f_4(x, a):
    return np.sin(a * x + 1) ** 2


def f_5(x):
    return x * np.tan(x)


def f_6(x, a, b):
    return (1 + a * x + b * (x ** 2)) / ((2/3) * (b + 3)) 


def f_7(x, a, b):
    return a + b * x


def f_8(x, a, b, c):
    return np.sin(a * x) + c * np.exp(b * x) + 1


def f_9(x, a, b):
    return np.exp(-(x - a) ** 2 / (2 * (b ** 2)))

def my_pdf(VAR, x):
    a = VAR

    pdf = f_1(x, a)

    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result


fname = 'Exam_2018_Prob1.txt'
data = np.loadtxt(fname)

z = data[:, 0]

a_bound = (-10, 0)
b_bound = (-10, 10)
c_bound = (4000, 8000)

n_bound = (0, None)
p_bound = (0, None)

mu_bound = (0, None)

data_0 = minimize(my_pdf, [1, ], args=(z), method='SLSQP',
                  bounds=(a_bound, ))


print(data_0)
x = np.arange(20, 27, 0.01)
y = f_1(x, -3)
plt.plot(x, y+0.2)
plt.hist(z, bins=200, normed=True)
plt.show()
binwidth = 0.1
n_bins = np.arange(min(data[:, 2]), max(data[:, 2]) + binwidth, binwidth)

# Chi2 calculator
# observed_values, bins, _ = plt.hist(data[:, 2], bins=n_bins)

# plt.show()
# We normalize by multiplyting the length of the data with the binwidth
# expected_values = poisson.pmf(bins, data_0.x[0]) * len(data) 

# print(observed_values[observed_values!=0])
# print(expected_values[expected_values!=0])
# print(chisquare(observed_values[observed_values!=0], f_exp=expected_values[expected_values!=0]))
# print('Threshold value ', chi2.isf(0.05, 18))


# x = np.arange(-1, 1, 0.01)
# y = f_6(x, data_0.x[0], data_0.x[1]) 
# plt.plot(x,y)
# plt.show()

