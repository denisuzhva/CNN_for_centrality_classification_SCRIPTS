import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



sums_shield = np.load('../../DATASETS/NA61_SHIELD/all/features_sum.npy').flatten()
sums_epos = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy').flatten()

counts_sh, ticks_sh, _ = plt.hist(sums_shield, bins=100)
counts_ep, ticks_ep, _ = plt.hist(sums_epos, bins=100)
max_count_sh = np.argmax(counts_sh)
max_count_ep = np.argmax(counts_ep)
max_tick_sh = ticks_sh[max_count_sh]
max_tick_ep = ticks_ep[max_count_ep]
shep_c = max_tick_sh / max_tick_ep
print(shep_c)

shep_cm = sums_shield.mean() / sums_epos.mean()

shep_c1 = 1.01 * shep_c
shep_c2 = 1.03 * shep_c
shep_c3 = 1.05 * shep_c
print(shep_c1)
print(shep_c2)
print(shep_c3)
print(shep_cm)

sums_epos_c0 = sums_epos * shep_c
sums_epos_c1 = sums_epos * shep_c1
sums_epos_c2 = sums_epos * shep_c2
sums_epos_c3 = sums_epos * shep_c3
sums_epos_cm = sums_epos * shep_cm

fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

axs[0].hist(sums_shield, bins=100, histtype='step', ec='b', density=True)
axs[0].hist(sums_epos_c0, bins=100, histtype='step', ec='r', density=True)
axs[0].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
axs[0].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
axs[0].set_title('Sum of measured energy (peak alignment: multiplier 0.806)')
axs[0].set_ylabel('density')
axs[0].set_xlabel('Measured energy')

axs[1].hist(sums_shield, bins=100, histtype='step', ec='b', density=True)
axs[1].hist(sums_epos_cm, bins=100, histtype='step', ec='r', density=True)
axs[1].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
axs[1].grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
axs[1].set_title('Sum of measured energy (center of mass alignment: multiplier 0.833)')
axs[1].set_xlabel('Measured energy')

plt.show()
