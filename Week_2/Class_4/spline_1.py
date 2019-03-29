import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

fname = 'DustLog_forClass.txt'

loaded_file = np.loadtxt(fname)

x = loaded_file[:, 0]
y = loaded_file[:, 1]

f = interp1d(x, y)
f1 = interp1d(x, y, kind='cubic')

x_new = np.linspace(min(x), max(x), len(x)*10000, endpoint=True)

plt.plot(x, y, 'k', marker='o', markersize='2', linestyle='none')
plt.plot(x, f(x), 'b', linestyle='--')
plt.plot(x_new, f1(x_new), 'r')

plt.show()

