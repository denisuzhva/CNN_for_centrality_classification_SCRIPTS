import numpy as np
import matplotlib.pyplot as plt



DATASET1 = 'EPOS_300prof'
DATASET2 = 'EPOS3_FIXprof'
DATA1_DIR = '../../DATASETS/NA61_{}/'.format(DATASET1)
DATA2_DIR = '../../DATASETS/NA61_{}/'.format(DATASET2)

dat1 = np.load(DATA1_DIR + 'features_sum.npy').astype(int)
dat2 = np.load(DATA2_DIR + 'features_sum.npy').astype(int)

dens = True
if dens:
    ylabel = 'density'
else:
    ylabel = 'entries'

fig, ax = plt.subplots()
ax.hist([dat1, dat2], bins=10, align='left', density=dens, label=['EPOS_300k', 'EPOS_100k'])
ax.legend(prop={'size': 10})
ax.set_xlabel('measured energy')
ax.set_ylabel(ylabel)

plt.show()