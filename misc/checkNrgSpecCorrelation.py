import numpy as np



DATASET = '../../DATASETS/NA61_EPOS_400prof/'
lab_nrg = np.load(DATASET + 'labels_nrg.npy').argmax(axis=1)
lab_spec = np.load(DATASET + 'labels_spec.npy').argmax(axis=1)

match_percent = np.sum(lab_nrg == lab_spec) / lab_nrg.size
print(match_percent)