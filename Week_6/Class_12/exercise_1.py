import numpy as np
import matplotlib.pyplot as plt


def test_statistic(N):

    N_bins = len(N)
    N_tot = np.sum(N)
    x = N[0]

    if x == 0:
        result = (2 * N_tot * np.log(N_bins / N_tot *
                                     N_tot / (N_bins - 1)))
    elif (N_tot - x) == 0:
        result = (2 * x * np.log(N_bins / N_tot * x))
    else:
        result = (2 * x * np.log(N_bins / N_tot * x) +
                  2 * (N_tot - x) * np.log(N_bins / N_tot *
                                           (N_tot - x) / (N_bins - 1)))
    return result


N = 1000
my_lambda = [0.1, 10, 1000]

for i, the_lambda in enumerate(my_lambda, 1):
    vars()['TS_' + str(i)] = np.array([])
    for _ in range(N):
        s = np.random.poisson(the_lambda, 100)
        temp = test_statistic(s)
        vars()['TS_' + str(i)] = np.append(vars()['TS_' + str(i)], temp)

# binwidth = 0.05
# n_bins = np.arange(min(s), max(s) + binwidth, binwidth)
plt.hist(TS_1, log=True, bins=100)
plt.figure()
plt.hist(TS_2, log=True, bins=100)
plt.figure()
plt.hist(TS_3, log=True, bins=100)
plt.show()
