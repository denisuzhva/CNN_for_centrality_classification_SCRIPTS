import numpy as np


dataset_name = 'NA61_better10_8x8'
all_dataset = np.load('../../DATASETS/{0}/all/all_dataset_flat.npy'.format(dataset_name))

all_sum = np.sum(all_dataset, axis=1)
all_sum = all_sum.reshape((all_sum.shape[0], 1))
print(all_sum.shape)

np.save('../../DATASETS/{0}/all/all_dataset_sum.npy'.format(dataset_name), all_sum)
