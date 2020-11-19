import numpy as np
import matplotlib.pyplot as plt



DATASET = 'EPOSfix'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)

stopvs = np.load(DATA_DIR + 'stopvs.npy')
stopvs = stopvs[np.logical_not(stopvs == 100500.)]

print(stopvs.max())
print(stopvs.min())

low_bound = -581.
high_bound = -576.
stopvs_low = stopvs[np.logical_and((stopvs >= low_bound), (stopvs <= high_bound))]

bin_width = 0.05
bins = np.arange(low_bound, high_bound + 0.05, 0.05)

fig, ax = plt.subplots()
ax.hist(stopvs_low, bins=bins, align='left')
ax.set_xlabel('Stop vertex')
ax.set_ylabel('entries')

plt.show()