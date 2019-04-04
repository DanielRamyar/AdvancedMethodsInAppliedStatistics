import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare, chi2


def expected_pdf(X, t):
    b, sigma = X

    temp1 = np.exp(-(2 * b * t - sigma ** 2) / (2 * b ** 2))
    temp2 = special.erf((b * t - sigma ** 2) / (np.sqrt(2) * b * sigma)) + 1
    temp3 = (2 * b)
    pdf = (temp1 * temp2) / temp3
    ln_pdf = np.log((pdf))
    result = np.sum(-ln_pdf)
    return result

def test_pdf(X, t):
    b, sigma = X

    temp1 = np.exp(-(2 * b * t - sigma ** 2) / (2 * b ** 2))
    temp2 = special.erf((b * t - sigma ** 2) / (np.sqrt(2) * b * sigma)) + 1
    temp3 = (2 * b)
    pdf = (temp1 * temp2) / temp3

    return pdf


fnames = 'ProblemSet2_Prob5_NucData.txt'
data = np.loadtxt(fnames)
data = data.reshape((100, 200))

h_0 = []
h_1 = []

for i in range(data.shape[0]):
    z = data[i, :]
    print(z.shape)

    result_1 = minimize(expected_pdf, [1, 1], args=(z), method='SLSQP',
                        bounds=((1, 1), (1e-10, None)))
    result_2 = minimize(expected_pdf, [1, 1], args=(z), method='SLSQP',
                        bounds=((1e-10, None), (1e-10, None)))

    h_0 = np.append(h_0, -result_1.fun)
    h_1 = np.append(h_1, -result_2.fun)

ratio = -2 * (h_0 - h_1)


print('Number of values in array larger than 2.706 is: %.2f'
      % len(ratio[ratio > 2.706]))

binwidth = 1
n_bins = np.arange(min(ratio), max(ratio) + binwidth, binwidth)
print(n_bins)
plt.figure()
observed_values, bins, _ = plt.hist(ratio, bins=n_bins,
                                    label='-2ln($\lambda$)')
plt.xlabel('-2ln($\lambda$)')
plt.ylabel('Count')
plt.legend()
plt.savefig('lnlikelihoods', dpi=200)

expected_values = [(special.erf(np.sqrt((i + 1) / 2)) -
                    special.erf(np.sqrt(i / 2))) * 100
                   for i in range(8)]

print(observed_values)
print(expected_values)
print(chisquare(observed_values, f_exp=expected_values))
print('Threshold value ', chi2.isf(0.05, 7))

##############################################################################
z = data.flatten()

result_1 = minimize(expected_pdf, [1, 1], args=(z), method='SLSQP',
                    bounds=((1.05, 1.05), (1e-10, None)))
result_2 = minimize(expected_pdf, [1, 1], args=(z), method='SLSQP',
                    bounds=((1e-10, None), (1e-10, None)))

h_0 = []
h_1 = []
h_0 = np.append(h_0, -result_1.fun)
h_1 = np.append(h_1, -result_2.fun)
ratio = -2 * (h_0 - h_1)

x = np.arange(-2.5, 10, 0.1)
plt.figure()
print(result_1.x, result_2.x)
plt.plot(x, test_pdf(result_1.x, x), label='$H_0$')
plt.plot(x, test_pdf(result_2.x, x), label='$H_1$')
plt.hist(z, density=True, bins=100, label='Data')
plt.legend()
plt.xlabel('t')
plt.savefig('data_h0h1', dpi=200)
plt.show()



