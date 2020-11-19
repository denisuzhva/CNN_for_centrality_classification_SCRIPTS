import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

from sklearn.linear_model import LinearRegression



mul = np.load('../../DATASETS/NA61_BeBeReal/all/mul.npy').flatten()
emeas = np.load('../../DATASETS/NA61_BeBeReal/all/features_sum.npy').flatten()

lin_fit = np.polyfit(emeas, mul, deg=1)
intercept_x = -lin_fit[1] / lin_fit[0]
print(lin_fit)
print(intercept_x)

x_fit = np.linspace(0, intercept_x)
y_fit = lin_fit[0] * x_fit + lin_fit[1]


plt.subplots(1, figsize=(7, 5), dpi=120)

plt.hist2d(emeas, mul, bins=(100, np.max(mul)), norm=LogNorm(), cmap='inferno')
plt.plot(x_fit, y_fit, '-r', label='Linear fit')
plt.legend(loc='upper right')

plt.grid()
plt.xlim((0, np.max(emeas)))
plt.ylim((0, np.max(mul)))
plt.ylabel('Multiplicity')
plt.xlabel('PSD sum')
plt.colorbar()
plt.show()