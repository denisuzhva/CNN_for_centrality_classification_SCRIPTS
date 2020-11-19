import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



DATASET_SIZE = 80000


## Prepare data
spec_num = np.load('../../DATASETS/NA61_SHIELD/all/spec24.npy')	# or spec_num
cen_modules = np.load('../../DATASETS/NA61_SHIELD/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_SHIELD/all/features_peripheral.npy')
spec_num = spec_num[:DATASET_SIZE]

p = np.random.permutation(DATASET_SIZE)
spec_num = spec_num[p]
cen_modules = cen_modules[p]
per_modules = per_modules[p]


## Make noise
r_disp = 0.5
r_mean = 1.

cen_noise = np.ones_like(cen_modules)
per_noise = np.ones_like(per_modules)
cen_noise_single = np.random.randn(1, 4, 4, 10, 1) * np.sqrt(r_disp) + r_mean
per_noise_single = np.random.randn(1, 4, 4, 10, 1) * np.sqrt(r_disp) + r_mean
cen_noise_act = np.repeat(cen_noise_single, DATASET_SIZE, axis=0)
per_noise_act = np.repeat(per_noise_single, DATASET_SIZE, axis=0)

cen_mod_noised = np.multiply(cen_modules, cen_noise_act)
per_mod_noised = np.multiply(per_modules, per_noise_act)
#cen_mod_noised = cen_modules
#per_mod_noised = per_modules
e_meas = np.sum(cen_mod_noised, (1, 2, 3, 4)).flatten() + np.sum(per_mod_noised, (1, 2, 3, 4)).flatten()


## Get bounds on train data
e_meas_sorted = np.sort(e_meas)
spec_num_sorted = np.sort(spec_num)

bound_perc = 23.9
e_meas_bound = e_meas_sorted[int(bound_perc*0.01*DATASET_SIZE)]
spec_num_bound = 3.5


## Calculate accuracy
tptn = 0
for ittt in range(DATASET_SIZE):
    if  (spec_num[ittt] <= spec_num_bound) and (e_meas[ittt] <= e_meas_bound) or \
        (spec_num[ittt] >= spec_num_bound) and (e_meas[ittt] >= e_meas_bound):
        tptn += 1

accuracy = float(tptn) / float(DATASET_SIZE)

print("Fine dots: %d \nAccuracy: %f" % (tptn, accuracy))


## Plotter
def plot2dHist():
    mat_sum_max = 25
    spec_num_max = 1200
    x_arrange_hor = np.arange(0, spec_num_max, 0.01)
    y_arrange_hor = np.full(spec_num_max*100, e_meas_bound)
    x_arrange_ver = np.full(mat_sum_max*100, spec_num_bound)
    y_arrange_ver = np.arange(0, mat_sum_max, 0.01)
    #plt.plot(spec_num, mat_sum, 'k.', markersize=0.8)
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec_num, e_meas, bins=300, cmap='inferno', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    cbar.set_label('entries', fontsize=18)
    plt.plot(x_arrange_hor, y_arrange_hor, color='blue')
    plt.plot(x_arrange_ver, y_arrange_ver, color='green')
    plt.xlabel('True energy', fontsize=18)
    plt.ylabel('Measured energy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axis([0, spec_num_max, 0, mat_sum_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnum.png')

#plot2dHist()
#plt.show()