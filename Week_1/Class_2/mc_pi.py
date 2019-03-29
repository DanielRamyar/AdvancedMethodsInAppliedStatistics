import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(142)

# Create my empty arrays
x = np.empty((1, 0))
y = np.empty((1, 0))
N = 1000

x_missed = np.empty((1, 0))
y_missed = np.empty((1, 0))

# Fill my arrays with random numbers
for i in range(N):

    x_random = random.random()
    y_random = random.random()
    if (np.sqrt(x_random**2 + y_random**2)) < 1:
        x = np.append(x, x_random)
        y = np.append(y, y_random)

    else:
        x_missed = np.append(x_missed, x_random)
        y_missed = np.append(y_missed, y_random)


# Plot the numbers
plt.plot(x, y, marker='o', linestyle='None')
plt.plot(x_missed, y_missed, marker='.', linestyle='None')

plt.show()

pi = len(x) / N * 4

print(pi)
