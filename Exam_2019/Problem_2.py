import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2, kstest


def uniform_CDF(x, min, max):
    return (x - min) / (max - min)


fname = 'Exam_2019_Problem2.txt'
data = np.loadtxt(fname, skiprows=2)

azimuth_angle = data[:, 0]
zenith_angle = np.cos(data[:, 1])

np.random.seed(421)
uniform_azimuth = np.random.uniform(0, 2 * np.pi, len(azimuth_angle))
uniform_zenith = np.random.uniform(-1, 1, len(zenith_angle))

###############################################################################
binwidth = 1
n_bins = np.arange(0, 2 * np.pi, binwidth)

observed_values, bins, _ = plt.hist(azimuth_angle,
                                    bins=n_bins,
                                    color='r',
                                    histtype='step',
                                    label='Azimuth Data',
                                    lw=2)
expected_values, bins, _ = plt.hist(uniform_azimuth,
                                    bins=n_bins,
                                    color='k',
                                    histtype='step',
                                    label='Monte Carlo Azimuth',
                                    lw=2)
plt.legend()
plt.show()

print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 1))

print('DF', len(expected_values) - 1)

print('KSTEST', kstest(azimuth_angle, lambda x: uniform_CDF(x, 0, 2 * np.pi)))


###############################################################################

binwidth = 0.5
n_bins = np.arange(-1, 1 + binwidth, binwidth)

observed_values1, bins, _ = plt.hist(zenith_angle,
                                     bins=n_bins,
                                     color='r',
                                     histtype='step',
                                     label='Zenith Data',
                                     lw=2)
expected_values1, bins, _ = plt.hist(uniform_zenith,
                                     bins=n_bins,
                                     color='k',
                                     histtype='step',
                                     label='Monte Carlo Zenith',
                                     lw=2)
plt.legend()
plt.show()

print(observed_values1)
print(expected_values1)
my_chi = chisquare(observed_values1,
                   f_exp=expected_values1)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values1) - 1))

print('DF', len(expected_values1) - 1)

print('KSTEST', kstest(zenith_angle, lambda x: uniform_CDF(x, -1, 1)))


###############################################################################
print('PRoblem B starts here')
print('H_A')
uniform_azimuth_20 = np.random.uniform(0.225*np.pi, 0.55 * np.pi, 20)
uniform_zenith_20 = np.cos(np.random.uniform(0.3*np.pi, np.pi, 20))

uniform_azimuth_80 = np.random.uniform(0, 2 * np.pi, 80)
uniform_zenith_80 = np.random.uniform(-1, 1, 80)


uniform_azimuth = np.append(uniform_azimuth_20, uniform_azimuth_80)
uniform_zenith = np.append(uniform_zenith_20, uniform_zenith_80)

binwidth = 1
n_bins = np.arange(0, 2 * np.pi, binwidth)

observed_values, bins, _ = plt.hist(azimuth_angle,
                                    bins=n_bins,
                                    color='r',
                                    histtype='step',
                                    label='Azimuth Data',
                                    lw=2)
expected_values, bins, _ = plt.hist(uniform_azimuth,
                                    bins=n_bins,
                                    color='k',
                                    histtype='step',
                                    label='Monte Carlo Azimuth 80/20',
                                    lw=2)
plt.legend()
plt.show()


print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 1))

print('DF', len(expected_values) - 1)

###############################################################################

binwidth = 0.5
n_bins = np.arange(-1, 1 + binwidth, binwidth)

observed_values, bins, _ = plt.hist(zenith_angle,
                                    bins=n_bins,
                                    color='r',
                                    histtype='step',
                                    label='Zenith Data',
                                    lw=2)
expected_values, bins, _ = plt.hist(uniform_zenith,
                                    bins=n_bins,
                                    color='k',
                                    histtype='step',
                                    label='Monte Carlo Zenith 80/20',
                                    lw=2)
plt.legend()
plt.show()


print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 1))

print('DF', len(expected_values) - 1)

###############################################################################
print('H_B')
uniform_azimuth_20 = np.random.uniform(0, np.pi, 15)
uniform_zenith_20 = np.cos(np.random.uniform(0.5*np.pi, np.pi, 15))

uniform_azimuth_80 = np.random.uniform(0, 2 * np.pi, 85)
uniform_zenith_80 = np.random.uniform(-1, 1, 85)


uniform_azimuth = np.append(uniform_azimuth_20, uniform_azimuth_80)
uniform_zenith = np.append(uniform_zenith_20, uniform_zenith_80)

binwidth = 1
n_bins = np.arange(0, 2 * np.pi, binwidth)

observed_values, bins, _ = plt.hist(azimuth_angle,
                                    bins=n_bins,
                                    color='r',
                                    histtype='step',
                                    label='Azimuth Data',
                                    lw=2)
expected_values, bins, _ = plt.hist(uniform_azimuth,
                                    bins=n_bins,
                                    color='k',
                                    histtype='step',
                                    label='Monte Carlo Azimuth 85/15',
                                    lw=2)
plt.legend()
plt.show()


print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 1))

print('DF', len(expected_values) - 1)

###############################################################################

binwidth = 0.5
n_bins = np.arange(-1, 1 + binwidth, binwidth)

observed_values, bins, _ = plt.hist(zenith_angle,
                                    bins=n_bins,
                                    color='r',
                                    histtype='step',
                                    label='Zenith Data',
                                    lw=2)
expected_values, bins, _ = plt.hist(uniform_zenith,
                                    bins=n_bins,
                                    color='k',
                                    histtype='step',
                                    label='Monte Carlo Zenith 85/15',
                                    lw=2)
plt.legend()
plt.show()


print(observed_values)
print(expected_values)
my_chi = chisquare(observed_values,
                   f_exp=expected_values)
print(my_chi)
print('Threshold value ', chi2.isf(0.05, len(expected_values) - 1))

print('DF', len(expected_values) - 1)
