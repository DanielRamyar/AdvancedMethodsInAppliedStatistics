import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.integrate as integrate


fname = 'SplineCubic.txt'

loaded_file = np.loadtxt(fname)

x = loaded_file[:, 0]
y = loaded_file[:, 1]

f = interp1d(x, y)
f1 = interp1d(x, y, kind='cubic')
f2 = interp1d(x, y, kind='quadratic')

x_new = np.linspace(min(x), max(x), len(x)*1000, endpoint=True)

plt.plot(x, y, 'k', marker='o', markersize='2', linestyle='none')
plt.plot(x, f(x), 'b', linestyle='--')
plt.plot(x_new, f1(x_new), 'r--')
plt.plot(x_new, f2(x_new), 'g--')

plt.show()

result = integrate.quad(lambda x: f(x), 10e-5, 0.01)
result_cubic = integrate.quad(lambda x: f1(x), 10e-5, 0.01)
result_quadradic = integrate.quad(lambda x: f2(x), 10e-5, 0.01)

print('Integral of Normal from 10e-5 to 0.01 is: %8.2f' % (result[0]))
print('Integral of Cubic from 10e-5 to 0.01 is: %8.2f' % (result_cubic[0]))
print('Integral of Quadratic from 10e-5 to 0.01 is: %8.2f' % (result_quadradic[0]))

result = integrate.quad(lambda x: f(x), 0.03, 0.1)
result_cubic = integrate.quad(lambda x: f1(x), 0.03, 0.1)
result_quadradic = integrate.quad(lambda x: f2(x), 0.03, 0.1)

print('Integral of Normal from 0.03 to 0.1 is: %8.2f' % (result[0]))
print('Integral of Cubic from 0.03 to 0.1 is: %8.2f' % (result_cubic[0]))
print('Integral of Quadratic from 0.03 to 0.1 is: %8.2f' % (result_quadradic[0]))