import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



mul = np.load('../../DATASETS/NA61_SHIELD/all/mul.npy').flatten()
emeas = np.load('../../DATASETS/NA61_SHIELD/all/features_sum.npy').flatten()

fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

scat = axs[0].scatter(emeas, mul, s=0.1, c='k')
hist = axs[1].hist2d(emeas, mul, bins=(100, np.arange(0, np.max(mul))+5), norm=LogNorm(), cmap='inferno')

axs[1].grid()

axs[0].set_ylabel('Multiplicity')
axs[0].set_xlabel('PSD sum')
axs[1].set_xlabel('PSD sum')

fig.colorbar(hist[3], ax=axs)

plt.show()