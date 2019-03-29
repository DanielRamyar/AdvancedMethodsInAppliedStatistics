import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import integrate
import random

random.seed(148)

# Deletes -99.99 from array and return flattened array
def remove_99(arr):
    temp = np.argwhere(arr == -99.99).flatten()
    return np.delete(arr.flatten(), temp)


# Load list of data into dictionary
def load_data(fnames):
    data = {name: np.loadtxt(name, skiprows=1) for name in fnames}
    return data


fnames = ['GlobalTemp_1.txt', 'GlobalTemp_2.txt']
data = load_data(fnames)

# Sanity check
print(data['GlobalTemp_1.txt'][6, 0])  # Should be 0.74
print(data['GlobalTemp_2.txt'][6, 0])  # Should be 1.07

# Remove -99.99 from row 8
data_row8_1997 = remove_99(data['GlobalTemp_1.txt'][6, None])
data_row8_2017 = remove_99(data['GlobalTemp_2.txt'][6, None])

x1 = np.linspace(-2, 4, 1000)
x2 = np.linspace(-2, 4, 1000)
kde1 = KernelDensity(kernel='epanechnikov',
                     bandwidth=0.4).fit(data_row8_1997[:, None])
kde2 = KernelDensity(kernel='epanechnikov',
                     bandwidth=0.4).fit(data_row8_2017[:, None])


def f_kde1(x):
    return np.exp((kde1.score_samples([[x]])))


def f_kde2(x):
    return np.exp((kde2.score_samples([[x]])))


# Remember score_samples return log(probability density) !!!!!
p1 = np.exp(kde1.score_samples(x1[:, None]))
p2 = np.exp(kde2.score_samples(x2[:, None]))

print('Integrating kde1 from -2 to 4 gives: %.2f' %
      integrate.quad(f_kde1, -2, 4)[0])
print('Integrating kde2 from -2 to 4 gives: %.2f' %
      integrate.quad(f_kde2, -2, 4)[0])
print('Integrating kde1 from -2 to 0 gives: %.2f' %
      integrate.quad(f_kde1, -2, 0)[0])
print('Integrating kde2 from -2 to 0 gives: %.2f' %
      integrate.quad(f_kde2, -2, 0)[0])


N = 1000
n_hit = 0
# Fill my arrays with random numbers

x = np.empty((1, 0))
y = np.empty((1, 0))
x_missed = np.empty((1, 0))
y_missed = np.empty((1, 0))

while n_hit < 1000:

    x_random = random.uniform(-1, 2)
    y_random = random.uniform(0, 1)
    if y_random < f_kde1(x_random):
        x = np.append(x, x_random)
        y = np.append(y, y_random)
        n_hit += 1
    else:
        x_missed = np.append(x_missed, x_random)
        y_missed = np.append(y_missed, y_random)

Likelihood_0 = np.exp((kde1.score_samples(x[:, None])))
Likelihood_1 = np.exp((kde2.score_samples(x[:, None])))

Likelihood_ratio = Likelihood_0 / Likelihood_1

print('Ratio is:', np.prod(Likelihood_ratio))

np.savetxt('ramyar_KDE_1000_samples.txt', x)

plt.figure(figsize=(7,4))
plt.plot(x1, p1, label='KDE Epanechnikov 1997 Data')
plt.plot(x2, p2, label='KDE Epanechnikov 2017 Data')
plt.plot(x, y, label='1000 Samples from 1997 KDE', marker='.',
         linestyle='None', markersize=2)
plt.legend()
plt.savefig('kdeplots', dpi=200)
# plt.show()

