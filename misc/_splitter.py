###################
# Concatenate two #
# .npy datasets   #
###################



import numpy as np



DATASET_full = '../../DATASETS/EPOS_415prof/'
DATASET_eduval = '../../DATASETS/EPOS_EV2prof/'
DATASET_test = '../../DATASETS/EPOS_T2prof/'

data_full = {}
data_agents = ['features_sum', 'features_central', 'features_peripheral',
               'mul', 'nrg24', 'spec24', 'labels_nrg', 'labels_spec']

for da in data_agents:
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')
    data_full[da] = np.load(DATASET_full + f'{da}.npy')

test_size = 100000 
eduval_size = data_full[data_agents[3]].shape[0] - test_size

for da in data_agents:
    np.save(DATASET_eduval + f'{da}.npy', data_full[da][:eduval_size])
    np.save(DATASET_test + f'{da}.npy', data_full[da][eduval_size:])
