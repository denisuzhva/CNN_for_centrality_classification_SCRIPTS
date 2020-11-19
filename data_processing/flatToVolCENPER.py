import numpy as np



dataset_path = '../../DATASETS/NA61_better10_8x8/all8/'
cen_flat = np.load(dataset_path + 'features_central_flat.npy')
per_flat = np.load(dataset_path + 'features_peripheral_flat.npy')

cen_nonflat = cen_flat.reshape(-1, 4, 4, 10, 1)
per_nonflat = per_flat.reshape(-1, 4, 4, 10, 1)

np.save(dataset_path + 'features_central.npy', cen_nonflat)
np.save(dataset_path + 'features_peripheral.npy', per_nonflat)
