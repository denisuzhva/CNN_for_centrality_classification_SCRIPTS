import numpy as np
import matplotlib.pyplot as plt



DATA_DIR = '../../DATASETS/NA61_EPOS/all/'
BOUNDS = 4
COLORS = 'brgy'

mul = np.load(DATA_DIR + 'mul.npy')
labels_arr = []
mul_first = []
for sb in range(BOUNDS):
    labels_arr.append(np.load(DATA_DIR + 'labels_spec{}.npy'.format(sb)).argmax(axis=1))
    mul_first.append(mul[labels_arr[sb] == 0])

fig, ax = plt.subplots()
for sb in range(BOUNDS):
    ax.hist(mul_first[3 - sb], bins=30, color=COLORS[sb], alpha=1, label=(3-sb))
ax.legend(loc='upper right')
ax.set_xlabel('multiplicity')
ax.set_ylabel('entries')

plt.show()

for sb in range(BOUNDS):
    print(mul_first[sb].size)

