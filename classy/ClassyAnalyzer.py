import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'




start_time = time.time()


e_meas = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy').flatten()
spec_nrg = np.load('../../DATASETS/NA61_EPOS/all/nrg24.npy').flatten()	# or spec_num

e_meas_sorted = np.sort(e_meas)
spec_nrg_sorted = np.sort(spec_nrg)

bound_perc = 24.9
#bound_perc = 23.9
e_meas_bound = e_meas_sorted[int(bound_perc*0.01*e_meas.size)]
spec_nrg_bound = spec_nrg_sorted[int(bound_perc*0.01*spec_nrg.size)]
print(spec_nrg_bound)

tp = 0
tptn = 0
fp = 0

for dot_iter in range(e_meas.size):
    if  (spec_nrg[dot_iter] <= spec_nrg_bound) and (e_meas[dot_iter] <= e_meas_bound) or \
        (spec_nrg[dot_iter] >= spec_nrg_bound) and (e_meas[dot_iter] >= e_meas_bound):
        tptn += 1


accuracy = float(tptn) / float(e_meas.size)
print(e_meas.size)

print("Fine dots: %d \nAccuracy: %f" % (tptn, accuracy))
print(e_meas_bound, spec_nrg_bound)


def plot2dHist():
    e_meas_max = 30 
    spec_nrg_max = 1200
    x_arrange_hor = np.arange(0, spec_nrg_max, 0.01)
    y_arrange_hor = np.full(spec_nrg_max*100, e_meas_bound)
    x_arrange_ver = np.full(e_meas_max*100, spec_nrg_bound)
    y_arrange_ver = np.arange(0, e_meas_max, 0.01)
    #plt.plot(spec_nrg, e_meas, 'k.', markersize=0.8)
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec_nrg, e_meas, bins=300, cmap='inferno', norm=LogNorm())
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
    plt.axis([0, spec_nrg_max, 0, e_meas_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnrg.png')


def plotProjX():
    plt.figure(figsize=(8, 6))
    plt.hist(spec_nrg, bins=np.arange(0, 1200, 1))
    plt.xlabel('True energy', fontsize=18)
    plt.ylabel('entries', fontsize=18)
    #plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    plt.axis([0, 1200, 0, 2000])


def plotProjY():
    plt.figure(figsize=(8, 6))
    plt.hist(e_meas, bins=np.arange(0, 25, 0.1))
    plt.xlabel('Measured energy', fontsize=18)
    plt.ylabel('entries', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    plt.axis([0, 25, 0, 2000])


plot2dHist()
#plotProjX()
plt.show()




print("--- %s seconds ---" % (time.time() - start_time))
