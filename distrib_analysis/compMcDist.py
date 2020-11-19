import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



sums_shield = np.load('../../DATASETS/NA61_SHIELD/all/features_sum.npy')
nrg_shield = np.load('../../DATASETS/NA61_SHIELD/all/nrg24.npy')
spec_shield = np.load('../../DATASETS/NA61_SHIELD/all/spec24.npy')

sums_epos = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy')
nrg_epos = np.load('../../DATASETS/NA61_EPOS/all/nrg24.npy')
spec_epos = np.load('../../DATASETS/NA61_EPOS/all/spec24.npy')

# Prepare bins
d = np.diff(np.unique(spec_shield)).min()
left_of_first_bin = spec_shield.min() - float(d)/2
right_of_last_bin = spec_shield.max() + float(d)/2

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
axs[0].hist(sums_shield, bins=100, histtype='step', ec='b', density=True)
axs[0].hist(sums_epos, bins=100, histtype='step', ec='r', density=True)
axs[0].set_title('Sum of measured energy')
axs[0].set_ylabel('density')

axs[1].hist(nrg_shield, bins=100, histtype='step', ec='b', density=True)
axs[1].hist(nrg_epos, bins=100, histtype='step', ec='r', density=True)
axs[1].set_title('Total forward energy')

axs[2].hist(spec_shield, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d), histtype='step', ec='b', density=True)
axs[2].hist(spec_epos, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d), histtype='step', ec='r', density=True)
axs[2].set_title('Total number of forward spectators')
axs[2].set_xticks(np.arange(np.min(spec_shield), np.max(spec_shield)+1, 1.0))

for i in range(3):
    axs[i].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    axs[i].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

plt.show()