import numpy as np



nrg = np.load('../../DATASETS/NA61_EPOS/all/nrg24.npy')
spec = np.load('../../DATASETS/NA61_EPOS/all/spec24.npy').astype(int)
lab_spec = np.load('../../DATASETS/NA61_EPOS/all/labels_spec.npy')

spec_bounds = np.arange(8)
bound_perc = np.size(spec[spec <= spec_bounds[7]]) / np.size(spec)
print(bound_perc)


print(lab_spec)