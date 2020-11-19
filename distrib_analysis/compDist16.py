import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



cen_shield = np.load('../../DATASETS/NA61_SHIELD/all/features_central.npy')
cen_epos = np.load('../../DATASETS/NA61_EPOS/all/features_central.npy')
sums_shield = np.load('../../DATASETS/NA61_SHIELD/all/features_sum.npy').flatten()
sums_epos = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy').flatten()

counts_sh, ticks_sh, _ = plt.hist(sums_shield, bins=100)
counts_ep, ticks_ep, _ = plt.hist(sums_epos, bins=100)
max_count_sh = np.argmax(counts_sh)
max_count_ep = np.argmax(counts_ep)
max_tick_sh = ticks_sh[max_count_sh]
max_tick_ep = ticks_ep[max_count_ep]
shep_constant = max_tick_sh / max_tick_ep
print(shep_constant)

cen_epos = cen_epos * shep_constant
sums_epos = sums_epos * shep_constant

mod_shield = np.sum(cen_shield, (3, 4))
mod_epos = np.sum(cen_epos, (3, 4))
fig, axs = plt.subplots(4, 4, figsize=(15, 15), dpi=120)
for i in range(4):
    for j in range(4):
        axs[i, j].hist(mod_shield[:, i, j], bins=100, histtype='step', ec='b', density=True)
        axs[i, j].hist(mod_epos[:, i, j], bins=100, histtype='step', ec='r', density=True)
        axs[i, j].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
        axs[i, j].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        axs[0, j].set_title('Sum of measured energy')
    axs[i, 0].set_ylabel('density')

plt.figure(figsize=(6, 4), dpi=120)
plt.hist(sums_shield, bins=100, histtype='step', ec='b', density=True)
plt.hist(sums_epos, bins=100, histtype='step', ec='r', density=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.title('Sum of measured energy')
plt.ylabel('density')

plt.show()
