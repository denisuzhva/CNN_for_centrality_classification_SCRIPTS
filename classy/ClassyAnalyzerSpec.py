import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'




start_time = time.time()


e_meas = np.load('../../DATASETS/NA61_EPOS/all/features_sum.npy').flatten()
spec_num = np.load('../../DATASETS/NA61_EPOS/all/spec24.npy').flatten()	# or spec_num

e_meas_sorted = np.sort(e_meas)
spec_num_sorted = np.sort(spec_num)

#bound_perc = 24.9
bound_perc = 23.9
e_meas_bound = e_meas_sorted[int(bound_perc*0.01*len(e_meas))]
spec_num_bound = 3.5


tp = 0
tptn = 0
fp = 0

for dot_iter in range(e_meas.size):
    if  (spec_num[dot_iter] <= spec_num_bound) and (e_meas[dot_iter] <= e_meas_bound) or \
        (spec_num[dot_iter] >= spec_num_bound) and (e_meas[dot_iter] >= e_meas_bound):
        tptn += 1


accuracy = float(tptn) / float(e_meas.size)
print(e_meas.size)

print("Fine dots: %d \nAccuracy: %f" % (tptn, accuracy))
print(e_meas_bound, spec_num_bound)


def plot2dHist():
    e_meas_max = 30
    spec_num_max = 7.5
    x_arrange_hor = np.arange(-0.5, spec_num_max, 0.01)
    y_arrange_hor = np.full(int(100*(spec_num_max+0.5)), e_meas_bound)
    x_arrange_ver = np.full(100*e_meas_max, spec_num_bound)
    y_arrange_ver = np.arange(0, e_meas_max, 0.01)
    #plt.plot(spec_num, e_meas, 'k.', markersize=0.8)
    binx = np.arange(10) - 0.5
    binx = binx.tolist()
    biny = np.arange(250) / 2
    biny = biny.tolist()
    plt.figure(figsize=(12, 7))
    plt.hist2d(spec_num, e_meas, bins=[binx, biny], norm=LogNorm(), cmap='inferno')
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
    plt.axis([-0.5, spec_num_max, 0, e_meas_max])
    #plt.savefig('../../../cnn-for-centrality-slides/Img/histspec.png')



def plotProjX():
    plt.hist(spec_num, bins=300)
    plt.xlabel('Number of spectators')

def plotProjY():
    plt.hist(e_meas, bins=300)
    plt.xlabel('Measured energy')


plot2dHist()
plt.show()




print("--- %s seconds ---" % (time.time() - start_time))
