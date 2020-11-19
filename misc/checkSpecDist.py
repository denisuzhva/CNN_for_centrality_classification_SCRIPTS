import numpy as np
import matplotlib.pyplot as plt



DATASET = 'EPOS_300k'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)

spec = np.load(DATA_DIR + 'spec24.npy').astype(int)

fig, ax = plt.subplots()
ax.hist(spec, bins=(np.max(spec)), align='left')
ax.set_xlabel('spec')
ax.set_ylabel('entries')

plt.show()