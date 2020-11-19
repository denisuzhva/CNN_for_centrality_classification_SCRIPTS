import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



## Calculate accuracy

DATA_DIR = '../../DATASETS/NA61_EPOS3prof/'
emeas = np.load(DATA_DIR + 'features_sum.npy').flatten()
nrg = np.load(DATA_DIR + 'nrg24.npy').flatten()
spec = np.load(DATA_DIR + 'spec24.npy').flatten()

emeas_sorted = np.sort(emeas)
nrg_sorted = np.sort(nrg)

spec_bound = 3
perc = spec[spec <= spec_bound].size / spec.size
nrg_bound = nrg_sorted[int(perc * nrg.size)]
emeas_bound = emeas_sorted[int(perc * emeas.size)]
print(perc)
print(nrg_bound)

acc_nrg = np.sum(np.logical_or(np.logical_and((nrg <= nrg_bound), (emeas <= emeas_bound)), np.logical_and((nrg > nrg_bound), (emeas > emeas_bound)))) / nrg.size
acc_spec = np.sum(np.logical_or(np.logical_and((spec <= spec_bound), (emeas <= emeas_bound)), np.logical_and((spec > spec_bound), (emeas > emeas_bound)))) / spec.size

print('NRG accuracy: %f, SPEC accuracy: %f' % (acc_nrg, acc_spec))


## Visualize

spec_bound_vis = spec_bound + 0.5

def plot2dHistNrg():
    emeas_max = int(np.max(1.1 * emeas))
    nrg_max = int(np.max(1.1 * nrg))
    x_arrange_hor = np.arange(0, nrg_max, 0.01)
    y_arrange_ver = np.arange(0, emeas_max, 0.01)
    y_arrange_hor = np.full(nrg_max*100, emeas_bound)
    x_arrange_ver = np.full(emeas_max*100, nrg_bound)
    plt.figure(figsize=(12, 7))
    plt.hist2d(nrg, emeas, bins=100, cmap='inferno', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    cbar.set_label('entries', fontsize=18)
    plt.plot(x_arrange_hor, y_arrange_hor, color='blue')
    plt.plot(x_arrange_ver, y_arrange_ver, color='green')
    plt.xlabel('Forward energy', fontsize=18)
    plt.ylabel('Measured energy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axis([0, nrg_max, 0, emeas_max])
    plt.tight_layout()
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnrg.png')
    

def plot2dHistSpec():
    e_meas_max = int(np.max(1.1 * emeas))
    spec_max = np.max(spec)+1
    x_arrange_hor = np.arange(-0.5, spec_max, 0.01)
    y_arrange_hor = np.full(int(100*(spec_max+0.5)), emeas_bound)
    x_arrange_ver = np.full(100*e_meas_max, spec_bound)
    y_arrange_ver = np.arange(0, e_meas_max, 0.01)
    #plt.plot(spec, e_meas, 'k.', markersize=0.8)
    binx = np.arange(spec_max) - 0.5
    binx = binx.tolist()
    biny = np.arange(250) / 2
    biny = biny.tolist()
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec, emeas, bins=[binx, biny], norm=LogNorm(), cmap='inferno')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    cbar.set_label('entries', fontsize=18)
    plt.plot(x_arrange_hor, y_arrange_hor, color='blue')
    plt.plot(x_arrange_ver, y_arrange_ver, color='green')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of spectators', fontsize=18)
    plt.ylabel('Measured energy', fontsize=18)
    plt.axis([-0.5, spec_max-1.5, 0, e_meas_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histspec.png') 


#plot2dHistNrg()
plot2dHistSpec()
plt.show()