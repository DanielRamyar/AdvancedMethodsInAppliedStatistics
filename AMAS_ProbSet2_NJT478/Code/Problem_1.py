import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

df = 1
x = np.linspace(0, 10, 1000)
y = chi2.pdf(x, df)

plt.style.use('bmh')

plt.plot(x, y, label='$\chi ^ 2$ pdf where df=1')
plt.ylabel('$PDF$ $(\chi ^ 2$)')
plt.xlabel('$\chi ^ 2$')
plt.legend()
plt.savefig('Figure_1', dpi=200)
# plt.show()