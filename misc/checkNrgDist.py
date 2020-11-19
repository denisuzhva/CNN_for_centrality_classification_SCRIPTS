import numpy as np
import matplotlib.pyplot as plt



DATASET = 'EPOS3prof'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)

spec = np.load(DATA_DIR + 'nrg24.npy').astype(int)

fig, ax = plt.subplots()
ax.hist(spec, bins=100, align='left')
ax.set_xlabel('forward energy')
ax.set_ylabel('entries')

plt.show()