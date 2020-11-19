import numpy as np
import matplotlib.pyplot as plt



DATASET = 'EPOS_400k'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)

pna = np.load(DATA_DIR + 'part_nrgs_all.npy').astype(int)
print(pna.max())

fig, ax = plt.subplots()
ax.hist(pna, bins=100, align='left')
ax.set_yscale('log')
ax.set_xlabel('energy')
ax.set_ylabel('entries')

plt.show()