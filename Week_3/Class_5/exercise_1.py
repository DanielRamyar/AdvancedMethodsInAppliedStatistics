import numpy as np
import matplotlib.pyplot as plt
import random


def my_func(x, alpha, beta):
    result = 1 + alpha * x + beta * x ** 2
    return result


def norm_factor(beta):
    # Calculated using wolfram maybe create function to do this?
    result = 0.571583 * beta + 1.9
    return result


alpha = 0.5
beta = 0.5

my_alphas = np.empty((1, 0))
my_betas = np.empty((1, 0))

random.seed(142)
n = 50
for _ in range(n):
    # Create my empty arrays
    x = np.empty((1, 0))
    N = 2000

    # Fill my arrays with random numbers
    for i in range(N):
        x_random = random.uniform(-0.95, 0.95)
        y_random = random.random()
        if y_random < my_func(x_random, alpha, beta) / norm_factor(beta):
            x = np.append(x, x_random)

    binwidth = 0.01
    alpha_scan = np.arange(0.1, 1 + binwidth, binwidth)
    beta_scan = np.arange(0.1, 1 + binwidth, binwidth)

    LLH = np.zeros((len(alpha_scan), len(beta_scan)))

    for i, scan_alpha in enumerate(alpha_scan, 0):
        for j, scan_beta in enumerate(beta_scan, 0):

            LLH[i, j] = np.sum(np.log(my_func(x, scan_alpha, scan_beta) /
                                      norm_factor(scan_beta)))

    ind = np.unravel_index(np.argmax(LLH, axis=None), LLH.shape)
    my_alphas = np.append(my_alphas, alpha_scan[ind[0]])
    my_betas = np.append(my_betas, alpha_scan[ind[1]])
# print(my_alphas, my_betas)

# Plot the numbers
plt.plot(my_alphas, my_betas, marker='o', linestyle='None')
plt.xlim(0, 1)
plt.ylim(0, 1)

binwidth = 0.05
n_bins = np.arange(min(my_alphas), max(my_alphas) + binwidth, binwidth)
plt.figure(2)
plt.axvline(x=np.mean(my_alphas)-np.std(my_alphas), ls = "--", color='#2ca02c')
plt.axvline(x=np.mean(my_alphas)+np.std(my_alphas), ls = "--", color='#2ca02c')
plt.hist(my_alphas, bins=n_bins)

binwidth = 0.05
n_bins = np.arange(min(my_betas), max(my_betas) + binwidth, binwidth)
plt.figure(3)
plt.hist(my_betas, bins=n_bins)
plt.axvline(x=np.mean(my_alphas)-np.std(my_alphas), ls = "--", color='#2ca02c')
plt.axvline(x=np.mean(my_alphas)+np.std(my_alphas), ls = "--", color='#2ca02c')

print('Mean of alpha is: %8.2f' % (np.mean(my_alphas)))
print('STD of alpha is: %8.2f' % (np.std(my_alphas)))
print('Mean of beta is: %8.2f' % (np.mean(my_betas)))
print('STD of beta is: %8.2f' % (np.std(my_betas)))

plt.show()
