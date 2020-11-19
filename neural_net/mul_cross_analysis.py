import numpy as np



### METADATA AND CMARAMETERS


## Dataset metadata

sh_name = 'SHIELD'
sh_size = 80000
sh_size_train = int(0.75 * sh_size)
sh_size_val = sh_size - sh_size_train
ep_name = 'EPOSprof'
ep_size = 98800
ep_size_train = int(0.75 * ep_size)
ep_size_val = ep_size - ep_size_train



## Boundary parameters

perc_sh = 23.9
perc_ep = 24.9

nrgbound_sh = 777.5
nrgbound_ep = 759.7421461



### LOAD DATA


## Load cross-test predictions

PRED_DIR = '../results/cross_test/'

sh_ep_nrg_pred = np.load(PRED_DIR + 'test_predictions_nrg_{}_{}_3.npy'.format(sh_name, ep_name)).reshape((ep_size, 2)).argmax(axis=1)
ep_sh_nrg_pred = np.load(PRED_DIR + 'test_predictions_nrg_{}_{}_3.npy'.format(ep_name, sh_name)).reshape((sh_size, 2)).argmax(axis=1)

sh_ep_spec_pred = np.load(PRED_DIR + 'test_predictions_spec_{}_{}_3.npy'.format(sh_name, ep_name)).reshape((ep_size, 2)).argmax(axis=1)
ep_sh_spec_pred = np.load(PRED_DIR + 'test_predictions_spec_{}_{}_3.npy'.format(ep_name, sh_name)).reshape((sh_size, 2)).argmax(axis=1)


## Also load validation predictions

sh_nrg_pred = np.load(PRED_DIR + 'test_predictions_nrg_{}_{}_3.npy'.format(sh_name, sh_name)).reshape((sh_size_val, 2)).argmax(axis=1)
ep_nrg_pred = np.load(PRED_DIR + 'test_predictions_nrg_{}_{}_3.npy'.format(ep_name, ep_name)).reshape((ep_size_val, 2)).argmax(axis=1)

sh_spec_pred = np.load(PRED_DIR + 'test_predictions_spec_{}_{}_3.npy'.format(sh_name, sh_name)).reshape((sh_size_val, 2)).argmax(axis=1)
ep_spec_pred = np.load(PRED_DIR + 'test_predictions_spec_{}_{}_3.npy'.format(ep_name, ep_name)).reshape((ep_size_val, 2)).argmax(axis=1)


## Load cross-labels and normal labels

DATA_DIR_SH = '../../DATASETS/NA61_{}/'.format(sh_name)
DATA_DIR_EP = '../../DATASETS/NA61_{}/'.format(ep_name)

ep_sh_nrg_lab = np.load(DATA_DIR_SH + 'labels_nrg_{}.npy'.format(ep_name)).argmax(axis=1)
sh_ep_nrg_lab = np.load(DATA_DIR_EP + 'labels_nrg_{}.npy'.format(sh_name)).argmax(axis=1)
sh_ep_nrg_lab = sh_ep_nrg_lab[:ep_size]

sh_nrg_lab = np.load(DATA_DIR_SH + 'labels_nrg.npy').argmax(axis=1)
ep_nrg_lab = np.load(DATA_DIR_EP + 'labels_nrg.npy').argmax(axis=1)
ep_nrg_lab = ep_nrg_lab[:ep_size]

sh_spec_lab = np.load(DATA_DIR_SH + 'labels_spec.npy').argmax(axis=1)
ep_spec_lab = np.load(DATA_DIR_EP + 'labels_spec.npy').argmax(axis=1)
ep_spec_lab = ep_spec_lab[:ep_size]


## Load multiplicity data

sh_mul = np.load(DATA_DIR_SH + 'mul.npy').flatten() 
ep_mul = np.load(DATA_DIR_EP + 'mul.npy').flatten() 
ep_mul = ep_mul[:ep_size]


## Load sums of measured energy for cut-based analysis

sh_sum = np.load(DATA_DIR_SH + 'features_sum.npy').flatten() 
ep_sum = np.load(DATA_DIR_EP + 'features_sum.npy').flatten() 
ep_sum = ep_sum[:ep_size]



### CALCULATE N AND OMEGA


## Calculator function

def n_omega_calc(lab, mul):
    mul_first_class = mul[lab == 0]
    am = mul_first_class.mean()
    nd = np.power(mul_first_class.std(), 2) / am
    am = np.round(am, 2)
    nd = np.round(nd, 2)
    return am, nd


## Prepre labels of CB

sh_sum_sorted = np.sort(sh_sum)
ep_sum_sorted = np.sort(ep_sum)

sumbound_sh = sh_sum_sorted[int(sh_sum.size * perc_sh / 100)]
sumbound_ep = ep_sum_sorted[int(ep_sum.size * perc_ep / 100)]

sh_cb_pred = np.zeros_like(sh_sum)
ep_cb_pred = np.zeros_like(ep_sum)
sh_cb_pred[sh_sum > sumbound_sh] = 1
ep_cb_pred[ep_sum > sumbound_ep] = 1


## N and Omega by CB

sh_cb_amnd = n_omega_calc(sh_cb_pred, sh_mul)
ep_cb_amnd = n_omega_calc(ep_cb_pred, ep_mul)


## N and Omega by the SHIELD-trained network

# Self-validation

sh_acc = np.sum(sh_nrg_pred == sh_nrg_lab[sh_size_train:]) / sh_nrg_pred.size
print(sh_acc)

sh_nrg_amnd = n_omega_calc(sh_nrg_pred, sh_mul[sh_size_train:])
sh_spec_amnd = n_omega_calc(sh_spec_pred, sh_mul[sh_size_train:])

# Cross-test on the EPOS dataset
sh_ep_nrg_amnd = n_omega_calc(sh_ep_nrg_pred, ep_mul)
sh_ep_spec_amnd = n_omega_calc(sh_ep_spec_pred, ep_mul)


## N and Omega by the EPOS-trained network

# Self-validation
ep_acc = np.sum(ep_nrg_pred == ep_nrg_lab[ep_size_train:]) / ep_nrg_pred.size
print(ep_acc)

ep_nrg_amnd = n_omega_calc(ep_nrg_pred, ep_mul[ep_size_train:])
ep_spec_amnd = n_omega_calc(ep_spec_pred, ep_mul[ep_size_train:])

# Cross-test on the EPOS dataset
ep_sh_nrg_amnd = n_omega_calc(ep_sh_nrg_pred, sh_mul)
ep_sh_spec_amnd = n_omega_calc(ep_sh_spec_pred, sh_mul)


## N and Omega by the labels

# Normal ones
sh_nrg_lab_amnd = n_omega_calc(sh_nrg_lab, sh_mul)
sh_spec_lab_amnd = n_omega_calc(sh_spec_lab, sh_mul)

ep_nrg_lab_amnd = n_omega_calc(ep_nrg_lab, ep_mul)
ep_spec_lab_amnd = n_omega_calc(ep_spec_lab, ep_mul)

# Cross-labels
sh_ep_nrg_lab_amnd = n_omega_calc(sh_ep_nrg_lab, ep_mul)
ep_sh_nrg_lab_amnd = n_omega_calc(ep_sh_nrg_lab, sh_mul)



### VISUALIZE


print('[<N>, scaled variance]')

print('**********************')
print('**********************')
print('Self-labels')

print('* SHIELD (e_true)')
print(sh_nrg_lab_amnd)
print('* SHIELD (n_spec)')
print(sh_spec_lab_amnd)

print('* EPOS (e_true)')
print(ep_nrg_lab_amnd)
print('* EPOS (n_spec)')
print(ep_spec_lab_amnd)

print('**********************')
print('**********************')
print('Cross-labels (e_true)')

print('* SHIELD with EPOS bnd')
print(ep_sh_nrg_lab_amnd)
print('* EPOS with SHIELD bnd')
print(sh_ep_nrg_lab_amnd)

print('**********************')
print('**********************')
print('Cut-based analysis')
print('* SHIELD')
print(sh_cb_amnd)
print('* EPOS')
print(ep_cb_amnd)

print('**********************')
print('**********************')
print('CNN self-validation')
print('* SHIELD e_true CNN')
print(sh_nrg_amnd)
print('* SHIELD n_spec CNN')
print(sh_spec_amnd)
print('* EPOS e_true CNN')
print(ep_nrg_amnd)
print('* EPOS n_spec CNN')
print(ep_spec_amnd)

print('**********************')
print('**********************')
print('CNN cross-test SH->EP')
print('* e_true')
print(sh_ep_nrg_amnd)
print('* n_spec')
print(sh_ep_spec_amnd)

print('**********************')
print('**********************')
print('CNN cross-test EP->SH')
print('* e_true')
print(ep_sh_nrg_amnd)
print('* n_spec')
print(ep_sh_spec_amnd)