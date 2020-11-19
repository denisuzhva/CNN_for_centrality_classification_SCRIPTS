import numpy as np



cen_modules = np.load('../../DATASETS/NA61_EPOS_cen_per_separated/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_EPOS_cen_per_separated/all/features_peripheral.npy')
e_meas = (np.sum(cen_modules, (1, 2, 3, 4)).flatten() + np.sum(per_modules, (1, 2, 3, 4)).flatten()).reshape((-1, 1))
np.save('../../DATASETS/NA61_EPOS_cen_per_separated/all/features_sum.npy', e_meas)