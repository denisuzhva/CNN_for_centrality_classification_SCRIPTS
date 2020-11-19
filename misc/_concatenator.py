###################
# Concatenate two #
# .npy datasets   #
###################



import numpy as np


DATASET1 = 'EPOS_317'
DATASET2 = 'EPOS_98'
DATASETf= 'EPOS_415'
DATA_DIR = ['../../DATASETS/NA61_{}/'.format(DATASET1), 
            '../../DATASETS/NA61_{}/'.format(DATASET2), 
            '../../DATASETS/NA61_{}/'.format(DATASETf)]

data_12 = [{}, {}]
data_agents = ['features_sum', 'features_central', 'features_peripheral',
               'mul', 'nrg24', 'spec24', 'labels_nrg', 'labels_spec']

for i in range(2):
    for data_agent in data_agents:
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
        data_12[i][data_agent] = np.load(DATA_DIR[i] + f'{data_agent}.npy')
d12_keys = data_12[0].keys()

for key in d12_keys:
    data_f = np.concatenate((data_12[0][key], data_12[1][key]), axis=0)
    np.save(DATA_DIR[2] + '{}.npy'.format(key), data_f)
