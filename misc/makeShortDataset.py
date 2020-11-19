import numpy as np



nrg = np.load('../../DATASETS/NA61_cen_per_separated/all/nrg24.npy')
spec = np.load('../../DATASETS/NA61_cen_per_separated/all/spec24.npy')

nrg = nrg[:80000]
spec = spec[:80000]

np.save('../../DATASETS/NA61_cen_per_separated/all/nrg24.npy', nrg)
np.save('../../DATASETS/NA61_cen_per_separated/all/spec24.npy', spec)