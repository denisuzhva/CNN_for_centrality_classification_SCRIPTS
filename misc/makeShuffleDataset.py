import numpy as np



DATASET = '../../DATASETS/NA61_EPOS_415prof/'
dl = []
dl.append(np.load(DATASET + 'features_central.npy'))
dl.append(np.load(DATASET + 'features_peripheral.npy'))
dl.append(np.load(DATASET + 'features_sum.npy'))
dl.append(np.load(DATASET + 'labels_nrg.npy'))
dl.append(np.load(DATASET + 'labels_spec.npy'))
dl.append(np.load(DATASET + 'mul.npy'))
dl.append(np.load(DATASET + 'nrg24.npy'))
dl.append(np.load(DATASET + 'spec24.npy'))

n_events = dl[0].shape[0]
p = np.random.permutation(n_events)

for i in range(len(dl)):
    dl[i] = dl[i][p].astype(np.float32)

np.save(DATASET + 'features_central.npy', dl[0])
np.save(DATASET + 'features_peripheral.npy', dl[1])
np.save(DATASET + 'features_sum.npy', dl[2])
np.save(DATASET + 'labels_nrg.npy', dl[3])
np.save(DATASET + 'labels_spec.npy', dl[4])
np.save(DATASET + 'mul.npy', dl[5])
np.save(DATASET + 'nrg24.npy', dl[6])
np.save(DATASET + 'spec24.npy', dl[7])
