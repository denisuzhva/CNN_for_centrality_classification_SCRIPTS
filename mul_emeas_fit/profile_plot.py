import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams["errorbar.capsize"] = 2



## Prepare data

mul = np.load('../../DATASETS/NA61_BeBeExper/all/mul.npy').flatten()
emeas = np.load('../../DATASETS/NA61_BeBeExper/all/features_sum.npy').flatten()


## Get the profile diagram

hist_breaks = np.histogram(emeas, bins=30)[1]
hb_size = hist_breaks.size

prof_mean = np.zeros((hb_size - 1), dtype=float)
prof_std = np.zeros_like(prof_mean)
mid_breaks = np.zeros_like(prof_mean)
prof_num = np.zeros_like(prof_mean)

for i in range(hb_size-1):
    prof_mean[i] = np.mean(mul[(emeas >= hist_breaks[i]) & (emeas < hist_breaks[i+1])])
    prof_std[i] = np.std(mul[(emeas >= hist_breaks[i]) & (emeas < hist_breaks[i+1])])
    mid_breaks[i] = np.mean([hist_breaks[i], hist_breaks[i+1]])
    prof_num[i] = mul[(emeas >= hist_breaks[i]) & (emeas < hist_breaks[i+1])].size

prof_eom = prof_std / np.sqrt(prof_num)


## Fit

cut_start = 0
cut_end = -4
prof_mean = prof_mean[cut_start:cut_end]
prof_std = prof_std[cut_start:cut_end]
mid_breaks = mid_breaks[cut_start:cut_end]
prof_num = prof_num[cut_start:cut_end]
prof_eom = prof_eom[cut_start:cut_end]

em1 = 2
em2 = 8

mb_lin = mid_breaks[em1:em2]
pm_lin = prof_mean[em1:em2]
ps_lin = prof_std[em1:em2]
pe_lin = prof_eom[em1:em2]
#mb_lin = mid_breaks
#pm_lin = prof_mean
#ps_lin = prof_std

cent_fit = np.polyfit(mb_lin, pm_lin, deg=1, w=1/pe_lin)
full_fit = np.polyfit(mid_breaks, prof_mean, deg=1, w=1/prof_eom)

cf_intercept = -cent_fit[1] / cent_fit[0]
ff_intercept = -full_fit[1] / full_fit[0]
print(cf_intercept)
print(ff_intercept)
print(cent_fit)
print(full_fit)

cf_x = np.linspace(0, np.max(emeas))
cf_y = cent_fit[0] * cf_x + cent_fit[1]

ff_x = np.linspace(0, np.max(emeas))
ff_y = full_fit[0] * ff_x + full_fit[1]


## Plot 

plt.errorbar(mid_breaks, prof_mean, prof_eom, fmt='ok', ms=2)
plt.plot(cf_x, cf_y, '-b', label='Linear fit (interval 250-750)')
plt.plot(ff_x, ff_y, '-r', label='Linear fit (full profile diagram)')
plt.legend(loc='upper right')
plt.xlabel('PSD sum')
plt.ylabel('Average multiplicity')
plt.xlim([0, np.max(emeas) + 1])
plt.ylim([0, np.max(prof_mean) + np.max(prof_std)])
plt.show()
