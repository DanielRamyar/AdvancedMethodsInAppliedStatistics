import numpy as np
import matplotlib.pyplot as plt
import nestle


def Likelihood(theta):
    my_function = np.sin(theta[0]) * np.sin(theta[1])
    return my_function


# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
def prior_transform(x):
    return 5 * np.pi * x  # - 5.0


x = np.arange(0, 5 * np.pi, 0.1)[np.newaxis]
y = np.arange(0, 5 * np.pi, 0.1)

plt.figure(1)
plt.imshow(Likelihood([x.T, y]), cmap='hot')
# plt.show()

# Run nested sampling.
result = nestle.sample(Likelihood, prior_transform, 2, npoints=1000, method='multi')


result.logz     # log evidence
result.logzerr  # numerical (sampling) error on logz
result.samples  # array of sample parameters
result.weights  # array of weights associated with each sample

plt.figure(2)
# plt.plot(result.samples[400:, 0], result.samples[400:, 1], '.')
plt.hist2d(result.samples[:, 0], result.samples[:, 1], weights=result.weights, bins=20)
plt.show()
