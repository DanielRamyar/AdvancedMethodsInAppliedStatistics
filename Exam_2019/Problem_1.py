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


def my_pdf1(VAR, x):
    a, b = VAR

    pdf = f_6(x, a, b)

    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result


def my_pdf2(VAR, k):
    mu = VAR

    pdf = poisson.pmf(k, mu)

    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result


def my_pdf3(VAR, x):
    a, b = VAR

    pdf = f_9(x, a, b)

    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result


fname = 'Exam_2019_Prob1.txt'
data = np.loadtxt(fname)


z0 = data[:, 0]
z1 = data[:, 1]
z2 = data[:, 2]

a_bound = (-10, 10)
b_bound = (-10, 10)
c_bound = (4000, 8000)

n_bound = (0, None)
p_bound = (0, None)

mu_bound = (0, None)

data_1 = minimize(my_pdf1, [1, 1], args=(z1), method='SLSQP',
                  bounds=(a_bound, b_bound))

data_2 = minimize(my_pdf2, [1, ], args=(z2), method='SLSQP',
                  bounds=(mu_bound, ))

data_3 = minimize(my_pdf3, [1, 1], args=(z2), method='SLSQP',
                  bounds=(a_bound, b_bound))


print(data_1)
print(data_2)
print(data_3)


###############################################################################
binwidth = 0.05
n_bins = np.arange(min(z1), max(z1) + binwidth, binwidth)

# Chi2 calculator
observed_values, bins, _ = plt.hist(z1, bins=n_bins)

# We normalize by multiplyting the length of the data with the binwidth
expected_values = f_6(bins[:-1], data_1.x[0], data_1.x[1]) * len(z1) * binwidth

print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values)-2))


plt.figure()
plt.hist(z1, bins=n_bins, normed=True, label='Data Column 2')
x = np.arange(-1, 1, 0.01)
y = f_6(x, data_1.x[0], data_1.x[1]) 
fit1_label = ('Fit: $1+ax+bx^2$, a=%.02f, b=%.02f \n$\chi^2=%.02f$' %
              (data_1.x[0], data_1.x[1], my_chi[0]))
plt.plot(x,y, label=fit1_label, color='r')
plt.legend()
plt.show()
###############################################################################
print('Data 2 starts here')
binwidth = 0.1
n_bins = np.arange(min(z2), max(z2) + binwidth, binwidth)

# Chi2 calculator
observed_values, bins, _ = plt.hist(z2, bins=n_bins, label='Data Column 3')
asdasd = np.arange(0, 21.1, 0.1)
expected_values = poisson.pmf(asdasd, data_2.x[0]) * len(z2)

print(observed_values[observed_values != 0][:-1])
print(expected_values[expected_values != 0][1:])

my_chi = chisquare(observed_values[observed_values != 0][:-1],
                f_exp=expected_values[expected_values != 0][1:])
print(my_chi)
print('Threshold value ', chi2.isf(0.05,
      len(expected_values[expected_values != 0][1:]) - 1))

x = np.arange(0, len(expected_values[expected_values != 0][1:]), 1)
fit1_label = ('Fit: Poisson distribution,\n$\mu$=%.02f, \n$\chi^2=%.02f$' %
              (data_1.x[0], my_chi[0]))
plt.plot(x,expected_values[expected_values != 0][1:], 'bo', label=fit1_label)
plt.legend()
plt.show()


###############################################################################




