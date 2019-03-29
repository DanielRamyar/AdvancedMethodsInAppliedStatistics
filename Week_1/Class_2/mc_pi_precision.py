import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(142)

# N is the number of throws per test
N = 100000
number_of_tests = 1
pi = np.empty((1, 0))
counter = 1
# Fill my arrays with random numbers
for i in range(number_of_tests):

    x = np.empty((1, 0))
    y = np.empty((1, 0))
    x_missed = np.empty((1, 0))
    y_missed = np.empty((1, 0))

    for i in range(N):

        x_random = random.random()
        y_random = random.random()
        if (np.sqrt(x_random**2 + y_random**2)) < 1:
            x = np.append(x, x_random)
            y = np.append(y, y_random)

        else:
            x_missed = np.append(x_missed, x_random)
            y_missed = np.append(y_missed, y_random)

        if counter in [10, 100, 1000, 10000, 100000]:
            pi = np.append(pi, len(x) / counter * 4)
        counter += 1


plt.plot([10, 100, 1000, 10000, 100000], pi, marker='o', linestyle='None')
plt.show()
