import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import random

fname = 'ProblemSet2_Problem2.txt'
data = np.loadtxt(fname, skiprows=2)

x_bottom_temp = data[0:12, 0]
y_bottom_temp = data[0:12, 1]

x_bottom_temp1 = data[37::, 0]
y_bottom_temp1 = data[37::, 1]

x_bottom = np.append(x_bottom_temp, x_bottom_temp1)
y_bottom = np.append(y_bottom_temp, y_bottom_temp1)

x_top = data[11:38, 0]
y_top = data[11:38, 1]

f_bottom = interpolate.interp1d(x_bottom, y_bottom)
xnew = np.linspace(0.3, 1.7, 1000)
ynew = f_bottom(xnew)


f_top = interpolate.interp1d(x_top, y_top)
xnew_1 = np.linspace(0.3, 1.7, 1000)
ynew_1 = f_top(xnew_1)


# N is the number of throws per test and A is area
N = 100000
number_of_tests = 1
A = np.empty((1, 0))

# Fill my arrays with random numbers
for i in range(number_of_tests):

    x = np.empty((1, 0))
    y = np.empty((1, 0))
    x_missed = np.empty((1, 0))
    y_missed = np.empty((1, 0))

    for i in range(N):

        x_random = random.uniform(0.3, 1.7)
        y_random = random.uniform(0.05, 0.35)
        if y_random < f_top(x_random):
            if y_random > f_bottom(x_random):
                x = np.append(x, x_random)
                y = np.append(y, y_random)
            else:
                x_missed = np.append(x_missed, x_random)
                y_missed = np.append(y_missed, y_random)

        else:
            x_missed = np.append(x_missed, x_random)
            y_missed = np.append(y_missed, y_random)

    A = np.append(A, len(x) / N * 4 * 5.2 ** 2)


plt.plot(x_top, y_top, 'k.')
plt.plot(x_bottom, y_bottom, 'k.')
plt.plot(xnew, ynew, 'k-')
plt.plot(xnew_1, ynew_1, 'k-')
plt.plot(x, y, color='silver', marker='.', linestyle='None', markersize=2, label='Accept')
plt.plot(x_missed, y_missed, color='black', linestyle='None', marker='.',
         markersize=2, label='Reject')
plt.title('Monte Carlo Area')
plt.legend(bbox_to_anchor=(0,1.0,1,0.2), loc="lower left")
plt.savefig('batman_logo.png', dpi=200)
# plt.show()

# Area 
A = (1.7 - 0.3) * (0.35 - 0.05)
print("Area: %.2f" % (A))

# Probality of hitting inside batman logo
p = len(x) / N

print("Probality hitting inside: %.2f%%" % (p))

# Area of logo
A_logo = A * p
print("Area: %.4f" % (A_logo))
