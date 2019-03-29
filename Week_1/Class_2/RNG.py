import random
import matplotlib.pyplot as plt
import numpy as np


# Initiate what seed i wanna use
random.seed(142)

# Create my empty arrays
x = np.empty((1, 0))
y = np.empty((1, 0))

# Fill my arrays with random numbers
for i in range(100):
    x = np.append(x, random.random())
    y = np.append(y, random.random())

# Plot the numbers
plt.plot(x, y, marker='o', linestyle='None')
plt.show()
