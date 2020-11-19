import numpy as np



DATA1_DIR = '../../DATASETS/NA61_{}/'.format('EPOS_415')
DATA2_DIR = '../../DATASETS/NA61_{}/'.format('EPOS_415prof')
emeas = np.load(DATA1_DIR + 'features_sum.npy')
cen = np.load(DATA1_DIR + 'features_central.npy')
per = np.load(DATA1_DIR + 'features_peripheral.npy')

ntr = 18.970710130278196
dtr = 23.17695067734108
shep_c = ntr / dtr 
emeas = emeas * shep_c
cen = cen * shep_c
per = per * shep_c

np.save(DATA2_DIR + 'features_sum.npy', emeas)
np.save(DATA2_DIR + 'features_central.npy', cen)
np.save(DATA2_DIR + 'features_peripheral.npy', per)

