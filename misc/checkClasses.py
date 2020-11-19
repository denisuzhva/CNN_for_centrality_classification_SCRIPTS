import numpy as np


DATASET = '../../DATASETS/EPOS_T2prof/'
spec_bounds = np.arange(7)

n_events = 0
for bound in spec_bounds:
    lab = np.load(DATASET + f'labels_spec{bound}.npy').argmax(axis=1)
    n_events = lab.size
    n_1_class = np.sum(lab == 0)
    n_2_class = np.sum(lab == 1)
    print('Bound: %i' % bound)
    print('# 1st class: %i' % n_1_class)
    print('# 2nd class: %i' % n_2_class)

