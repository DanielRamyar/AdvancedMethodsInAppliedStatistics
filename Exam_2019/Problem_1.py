import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare, chi2, binom, poisson
import random
from scipy import integrate


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
    return (np.sin(a * x) + c * np.exp(b * x) + 1)


def f_9(x, a, b):
    normz = (np.sqrt(2 * np.pi) * np.abs(b))
    my_func = np.exp(-(x - a) ** 2 / (2 * (b ** 2))) / normz
    return my_func


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
    a, b, c = VAR

    normz = integrate.quad(f_8, 20, 27, args=(a, b, c))

    pdf = f_8(x, a, b, c) / normz[0]

    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result


fname = 'Exam_2019_Prob1.txt'
data = np.loadtxt(fname)


z0 = data[:, 0]
z1 = data[:, 1]
z2 = data[:, 2]

a_bound = (0, 10)
b_bound = (-10, 10)
c_bound = (4000, 8000)

n_bound = (0, None)
p_bound = (0, None)

mu_bound = (0, None)

data_0 = minimize(my_pdf3, [4, -0.1, 4150], args=(z0), method='SLSQP',
                  bounds=(a_bound, b_bound, c_bound))

data_1 = minimize(my_pdf1, [1, 1], args=(z1), method='SLSQP',
                  bounds=(a_bound, b_bound))

data_2 = minimize(my_pdf2, [1, ], args=(z2), method='SLSQP',
                  bounds=(mu_bound, ))


print(data_0)
print(data_1)
print(data_2)


###############################################################################
binwidth = 0.05
n_bins = np.arange(min(z1), max(z1) + binwidth, binwidth)

# Chi2 calculator
observed_values, bins, _ = plt.hist(z1, bins=n_bins)
plt.close()
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
plt.savefig('prob1_Column2.pdf')
# plt.show()
###############################################################################
print('Data 2 starts here')
binwidth = 0.1
n_bins = np.arange(min(z2), max(z2) + binwidth, binwidth)

# Chi2 calculator
observed_values, bins, _ = plt.hist(z2, bins=n_bins, label='Data Column 3')
asdasd = np.arange(0, 21.1, 0.1)
expected_values = poisson.pmf(asdasd, data_2.x[0]) * len(z2)
plt.close()
print(observed_values[observed_values != 0][:-1])
print(expected_values[expected_values != 0][1:])

my_chi = chisquare(observed_values[observed_values != 0][:-1],
                   f_exp=expected_values[expected_values != 0][1:])
print(my_chi)
print('Threshold value ', chi2.isf(0.05,
      len(expected_values[expected_values != 0][1:]) - 1))

plt.hist(z2, bins=n_bins+0.05, label='Data Column 3')
x = np.arange(0, len(expected_values[expected_values != 0][1:]), 1)
fit1_label = ('Fit: Poisson distribution,\n$\mu$=%.02f, \n$\chi^2=%.02f$' %
              (data_2.x[0], my_chi[0]))
plt.plot(x, expected_values[expected_values != 0][1:], 'ro', label=fit1_label)
plt.legend()
plt.savefig('prob1_Column3.pdf')

# plt.show()


###############################################################################
plt.figure()
binwidth = 0.1
n_bins = np.arange(min(z0), max(z0) + binwidth, binwidth)

# Chi2 calculator
observed_values, bins, _ = plt.hist(z0, bins=n_bins)

normz = integrate.quad(f_8, 20, 27, args=(data_0.x[0], data_0.x[1], data_0.x[2]))
# # We normalize by multiplyting the length of the data with the binwidth
expected_values = (f_8(bins[:-1], data_0.x[0], data_0.x[1], data_0.x[2]) / normz[0] *
                   len(z0) * binwidth)
plt.close()
print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 3))

plt.figure()

plt.hist(z0, bins=n_bins, normed=True, label='Data Column 2')
x = np.arange(20, 27, 0.01)
y = f_8(x, data_0.x[0], data_0.x[1], data_0.x[2])/normz[0]


fit1_label = ('Fit: $sin(ax)+ce^{bx}+1$,\na=%.02f, b=%.02f, c=%.05f \n$\chi^2=%.02f$' %
              (data_0.x[0], data_0.x[1], data_0.x[2], my_chi[0]))
plt.plot(x,y, color='r', label=fit1_label)
plt.legend()
plt.savefig('prob1_Column1.pdf')

# plt.show()

