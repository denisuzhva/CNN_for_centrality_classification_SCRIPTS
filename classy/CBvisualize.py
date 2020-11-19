import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



emeas = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy').flatten()
nrg = np.load('../../DATASETS/NA61_EPOS/all/nrg24.npy').flatten()
spec = np.load('../../DATASETS/NA61_EPOS/all/spec24.npy').flatten()

emeas_sorted = np.sort(emeas)
nrg_sorted = np.sort(nrg)

spec_bounds = np.arange(7)
nrg_bounds = np.zeros_like(spec_bounds, dtype=float)
emeas_bounds = np.zeros_like(spec_bounds, dtype=float)

for idx, spec_bound in enumerate(spec_bounds):
    perc = spec[spec <= spec_bound].size / spec.size
    nrg_bounds[idx] = nrg_sorted[int(perc * nrg.size)]
    emeas_bounds[idx] = emeas_sorted[int(perc * emeas.size)]

spec_bounds_vis = spec_bounds + 0.5
colors = 'bgrcmky'

def plot2dHistNrg():
    emeas_max = int(np.max(1.1 * emeas))
    nrg_max = int(np.max(1.1 * nrg))
    x_arrange_hor = np.arange(0, nrg_max, 0.01)
    y_arrange_ver = np.arange(0, emeas_max, 0.01)
    
    plt.figure(figsize=(12, 7))
    plt.hist2d(nrg, emeas, bins=300, cmap='inferno', norm=LogNorm())

    for idx, _ in enumerate(spec_bounds):
        y_arrange_hor = np.full(nrg_max*100, emeas_bounds[idx])
        x_arrange_ver = np.full(emeas_max*100, nrg_bounds[idx])
        plt.plot(x_arrange_hor, y_arrange_hor, color=colors[idx])
        plt.plot(x_arrange_ver, y_arrange_ver, color=colors[idx])
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('entries', fontsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    cbar.set_label('entries', fontsize=18)
    
    plt.xlabel('True energy', fontsize=18)
    plt.ylabel('Measured energy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axis([0, nrg_max, 0, emeas_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnrg.png')


def plot2dHistSpec():
    emeas_max = int(np.max(1.1 * emeas))
    spec_max = int(np.max(spec))
    x_arrange_hor = np.arange(0, spec_max, 0.01)
    y_arrange_ver = np.arange(0, emeas_max, 0.01)
    
    binx = np.arange(10) - 0.5
    binx = binx.tolist()
    biny = np.arange(250) / 2
    biny = biny.tolist()
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec, emeas, bins=[binx, biny], cmap='inferno', norm=LogNorm())

    for idx, _ in enumerate(spec_bounds):
        y_arrange_hor = np.full(spec_max*100, emeas_bounds[idx])
        x_arrange_ver = np.full(emeas_max*100, spec_bounds_vis[idx])
        plt.plot(x_arrange_hor, y_arrange_hor, color=colors[idx])
        plt.plot(x_arrange_ver, y_arrange_ver, color=colors[idx])
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('entries', fontsize=18)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    cbar.set_label('entries', fontsize=18)
    
    plt.xlabel('Number of spectators', fontsize=18)
    plt.ylabel('Measured energy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axis([0, spec_max, 0, emeas_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histnrg.png'


plot2dHistNrg()
#plot2dHistSpec()
plt.show()