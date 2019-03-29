import numpy as np
import matplotlib.pyplot as plt




for i in np.arange(0, 11, 0.05):
    my_lambda = i
    s = np.random.poisson(my_lambda, 100)


y, x, _ = plt.hist(s)
print(x)
print(np.sum(y))
plt.show()