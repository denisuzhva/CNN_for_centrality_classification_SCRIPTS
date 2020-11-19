import numpy as np


cen_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_peripheral.npy')
print(cen_modules.shape)
print(per_modules.shape)