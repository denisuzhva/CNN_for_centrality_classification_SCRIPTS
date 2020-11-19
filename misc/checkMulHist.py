import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



DATASET = 'EPOS_300k'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)
BOUNDS = 7
COLORS = 'brgycmk'

mul = np.load(DATA_DIR + 'mul.npy')
nrg = np.load(DATA_DIR + 'nrg24.npy')
spec = np.load(DATA_DIR + 'spec24.npy')
labels_arr = []
mul_first = []
nrg_first = []
spec_first = []
for sb in range(BOUNDS):
    labels_arr.append(np.load(DATA_DIR + 'labels_spec{}.npy'.format(sb)).argmax(axis=1))
    mul_first.append(mul[labels_arr[sb] == 0])
    nrg_first.append(nrg[labels_arr[sb] == 0])
    spec_first.append(spec[labels_arr[sb] == 0])

# mul hist
def plotMul():
    _, ax = plt.subplots()
    for sb in range(BOUNDS):
        max_mul = mul_first[BOUNDS - 1 - sb].max()
        min_mul = mul_first[BOUNDS - 1 - sb].min()
        ax.hist(mul_first[BOUNDS - 1 - sb], bins=max_mul-min_mul+1, color=COLORS[sb], alpha=1, label=(BOUNDS - 1 - sb), align='left')
    ax.set_title(DATASET)
    ax.legend(loc='upper right')
    ax.set_xlabel('multiplicity')
    ax.set_ylabel('entries')
    ax.set_yscale('log')

# mul against nrg
def plotMulNrg():
    rows = 2
    cols = 2
    space = 0.4
    fig, ax = plt.subplots(rows, cols)
    fig.subplots_adjust(wspace=space, hspace=space)
    for i in range(rows):
        for j in range(cols):
            n = i * cols + j
            binx = 30
            biny = mul_first[n].max()+1
            ax[i, j].hist2d(nrg_first[n], mul_first[n], bins=[binx, biny], norm=LogNorm(), cmap='inferno')
            ax[i, j].set_title('bound nspec %i' % i)
            if i == rows-1:
                ax[i, j].set_xlabel('eforw')
            if j == 0:
                ax[i, j].set_ylabel('mul')

# mul against spec
def plotMulSpec():
    rows = 2
    cols = 2
    space = 0.4
    fig, ax = plt.subplots(rows, cols)
    fig.subplots_adjust(wspace=space, hspace=space)
    for i in range(rows):
        for j in range(cols):
            n = i * cols + j
            binx = (np.arange((spec_first[n].max()+2)) - 0.5).tolist()
            biny = mul_first[i*cols+j].max()+1
            ax[i, j].hist2d(spec_first[n], mul_first[n], bins=[binx, biny], norm=LogNorm(), cmap='inferno')
            ax[i, j].set_title('bound nspec %i' % i)
            if i == rows-1:
                ax[i, j].set_xlabel('spec')
            if j == 0:
                ax[i, j].set_ylabel('mul')

plotMul()
plt.show()

#for sb in range(BOUNDS):
#    print(mul_first[sb].size)