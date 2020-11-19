import numpy as np



DATA_DIR = '../../DATASETS/NA61_EPOS_400prof/'
mul = np.load(DATA_DIR + 'mul.npy')
#labSHIELD = np.load(DATA_DIR + 'labels_nrg_SHIELD.npy')

print(mul.shape)
#print(labSHIELD.dtype)