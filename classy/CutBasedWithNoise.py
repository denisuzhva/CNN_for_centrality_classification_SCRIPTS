import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



TRAIN_SIZE = 60000
TEST_SIZE = 20000
DATASET_SIZE = TRAIN_SIZE + TEST_SIZE


## Prepare data
spec_nrg = np.load('../../DATASETS/NA61_SHIELD/all/nrg24.npy')	# or spec_num
cen_modules = np.load('../../DATASETS/NA61_SHIELD/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_SHIELD/all/features_peripheral.npy')
print(spec_nrg.shape)
print(cen_modules.shape)
spec_nrg = spec_nrg[:DATASET_SIZE]

p = np.random.permutation(DATASET_SIZE)
spec_nrg = spec_nrg[p]
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
cen_noise[:, :, :, :, :] = cen_noise_act
per_noise[:, :, :, :, :] = per_noise_act

cen_mod_noised = np.multiply(cen_modules, cen_noise)
per_mod_noised = np.multiply(per_modules, per_noise)
#cen_mod_noised = cen_modules
#per_mod_noised = per_modules
e_meas = np.sum(cen_mod_noised, (1, 2, 3, 4)).flatten() + np.sum(per_mod_noised, (1, 2, 3, 4)).flatten()


## Get bounds on train data
e_meas_train = e_meas[:]
spec_nrg_train = spec_nrg[:]
e_meas_train_sorted = np.sort(e_meas_train)
spec_nrg_train_sorted = np.sort(spec_nrg_train)

bound_perc = 23.9
e_meas_bound_train = e_meas_train_sorted[int(bound_perc*0.01*DATASET_SIZE)]
spec_nrg_bound_train = spec_nrg_train_sorted[int(bound_perc*0.01*DATASET_SIZE)]
print(spec_nrg_bound_train)


## Calculate accuracy
tptn = 0
e_meas_test = e_meas[:]
spec_nrg_test = spec_nrg[:]
for ittt in range(DATASET_SIZE):
    if  (spec_nrg_test[ittt] <= spec_nrg_bound_train) and (e_meas_train[ittt] <= e_meas_bound_train) or \
        (spec_nrg_test[ittt] >= spec_nrg_bound_train) and (e_meas_train[ittt] >= e_meas_bound_train):
        tptn += 1

accuracy = float(tptn) / float(DATASET_SIZE)

print("Fine dots: %d \nAccuracy: %f" % (tptn, accuracy))


## Plotter
def plot2dHist():
    mat_sum_max = 25
    spec_nrg_max = 1200
    x_arrange_hor = np.arange(0, spec_nrg_max, 0.01)
    y_arrange_hor = np.full(spec_nrg_max*100, e_meas_bound_train)
    x_arrange_ver = np.full(mat_sum_max*100, spec_nrg_bound_train)
    y_arrange_ver = np.arange(0, mat_sum_max, 0.01)
    #plt.plot(spec_nrg, mat_sum, 'k.', markersize=0.8)
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec_nrg_test, e_meas_test, bins=300, cmap='inferno', norm=LogNorm())
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
    plt.axis([0, spec_nrg_max, 0, mat_sum_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnrg.png')

plot2dHist()
plt.show()