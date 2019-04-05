import numpy as np
import matplotlib.pyplot as plt
import nestle
import scipy 

plt.style.use('bmh')


def Likelihood(theta):
    mu = 0.68
    sigma = 0.2
    temp1 = np.cos(theta[0]) * np.cos(theta[1])
    temp2 = 1 / (sigma * np.sqrt(2 * np.pi))
    temp3 = np.exp(- (theta[2] - mu) ** 2 / (2 * sigma ** 2))
    temp4 = np.cos(theta[0] / 2)

    my_function = 3 * (temp1 + temp2 * temp3 * temp4 + 3)

    return my_function


# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
def prior_transform(x):
    return 7 * np.pi * x   # - 5.0


x = np.arange(0, 7 * np.pi, 0.1)[np.newaxis]
y = np.arange(0, 7 * np.pi, 0.1)
z = np.arange(0, 3, 0.1)


# plt.figure(1)
# plt.imshow(Likelihood([x.T, y]), cmap='hot')
# plt.show()

# Run nested sampling.
result = nestle.sample(Likelihood, prior_transform, 3, npoints=1000, method='multi')


result.logz     # log evidence
result.logzerr  # numerical (sampling) error on logz
result.samples  # array of sample parameters
result.weights  # array of weights associated with each sample

plt.figure(2)
plt.plot(result.samples[400:, 0], result.samples[400:, 2], '.')
plt.xlabel('$\\theta_{1}$')
plt.ylabel('$\\theta_{3}$')
plt.ylim(0, 3) 
# # plt.hist2d(result.samples[:, 0], result.samples[:, 1], weights=result.weights, bins=20)
plt.show()

print('Best theta1', result.samples[-1, 0])
print('Best theta2', result.samples[-1, 1])
print('Best theta3', result.samples[-1, 2])

binwidth = 0.1
mu_scan = np.arange(0.05, 7 * np.pi + binwidth, binwidth)
sigma_scan = np.arange(0.05, 7 * np.pi + binwidth, binwidth)


LLH = np.zeros((len(mu_scan), len(sigma_scan)))

best1 = 12.349165650011875
best2 = 6.781712075143268
best3 = 0.7898809313896807

for i, mu in enumerate(mu_scan, 0):
    for j, sigma in enumerate(sigma_scan, 0):
        theta = [mu, sigma, best3]
        LLH[i, j] = np.sum(np.log(Likelihood(theta)))


plt.imshow(LLH, cmap='hot', origin='lower', extent=(0, 7 * np.pi, 0, 7 * np.pi))
plt.xlabel('$\\theta_{1}$')
plt.ylabel('$\\theta_{2}$')
plt.show()

binwidth = 0.1
mu_scan = np.arange(0.05, 7 * np.pi + binwidth, binwidth)
sigma_scan = np.arange(0.05, 3 + binwidth, binwidth)


LLH = np.zeros((len(mu_scan), len(sigma_scan)))


for i, mu in enumerate(mu_scan, 0):
    for j, sigma in enumerate(sigma_scan, 0):
        theta = [mu, best2, sigma_scan]
        LLH[i, j] = np.sum(np.log(Likelihood(theta)))


plt.imshow(LLH.T, cmap='hot', origin='lower', extent=(0, 7 * np.pi, 0, 3))
plt.xlabel('$\\theta_{1}$')
plt.ylabel('$\\theta_{3}$')
plt.show()

