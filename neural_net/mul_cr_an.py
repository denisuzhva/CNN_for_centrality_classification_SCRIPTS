import numpy as np



### MISC


## Dataset metadata

sh_name = 'SHIELD'
ep_name = 'EPOSprof'
DN = [sh_name, ep_name]
nrg_name = 'nrg'
spec_name = 'spec'
DM = [nrg_name, spec_name]
sh_size = 80000
ep_size = 98800
D_SIZES = {sh_name: sh_size, ep_name: ep_size}
D_S_TRAIN = {DN[i]: int(0.75 * dsize) for i, dsize in enumerate(D_SIZES.values())}
D_S_TEST = {DN[i]: D_SIZES[DN[i]] - D_S_TRAIN[DN[i]] for i, _ in enumerate(D_SIZES.values())}


## Boundary parameters

cperc = {sh_name: 23.9, ep_name: 24.9}
nrg_bound = {sh_name: 777.5, ep_name: 759.7421461}


## Directories
PRED_DIR = '../results/cross_test/'
DATA_DIR = {}
for dn in DN:
    DATA_DIR[dn] = '../../DATASETS/NA61_{}/'.format(dn)


## Calculator function

def n_omega_calc(lab, mul):
    mul_first_class = mul[lab == 0]
    am = mul_first_class.mean()
    nd = np.power(mul_first_class.std(), 2) / am
    am = np.round(am, 2)
    nd = np.round(nd, 2)
    return am, nd



### LOAD DATA


## Load predictions

nn_pred = {}
for dn_model in DN:
    nn_pred[dn_model] = {}
    for dn_test in DN:
        nn_pred[dn_model][dn_test] = {}
        for mode in DM:
            nn_pred[dn_model][dn_test][mode] = np.load(PRED_DIR + \
                'test_predictions_{}_{}_{}_3.npy'.format(mode, dn_model, dn_test)).reshape((-1, 2)).argmax(axis=1)
        

## Load labels

labels = {}
for dn_model in DN:
    labels[dn_model] = {}
    for dn_test in DN:
        labels[dn_model][dn_test] = {}
        for mode in DM:
            labels[dn_model][dn_test][mode] = np.load(PRED_DIR + \
                'labels_{}_{}_{}.npy'.format(mode, dn_model, dn_test)).argmax(axis=1)


## Load simulation data

# multiplicity
multiplicity = {}
for dn in DN:
    multiplicity[dn] = np.load(DATA_DIR[dn] + 'mul.npy').flatten() 

# sums of measured energy
esum = {}
for dn in DN:
    esum[dn] = np.load(DATA_DIR[dn] + 'features_sum.npy').flatten() 


## CB labels

cb_pred = {}
for dn in DN:
    sumbound = np.sort(esum[dn])[int(D_SIZES[dn] * cperc[dn] / 100.)]
    cb_pred[dn] = np.zeros((D_SIZES[dn]))
    cb_pred[dn][esum[dn] > sumbound] = 1



### CALCULATE <N> AND OMEGA


## CBA

cb_amnd = {}
for dn in DN:
    cb_amnd[dn] = n_omega_calc(cb_pred[dn], multiplicity[dn])


## CNN

nn_amnd = {}
for dn_model in DN:
    nn_amnd[dn_model] = {}
    for dn_test in DN:
        nn_amnd[dn_model][dn_test] = {}
        for mode in DM:
            if dn_model == dn_test:
                cur_mul = multiplicity[dn_test][D_S_TRAIN[dn_test]:]
            else:
                cur_mul = multiplicity[dn_test]
            nn_amnd[dn_model][dn_test][mode] = n_omega_calc(
                nn_pred[dn_model][dn_test][mode], cur_mul)


## Label

lab_amnd = {}
for dn_model in DN:
    lab_amnd[dn_model] = {}
    for dn_test in DN:
        lab_amnd[dn_model][dn_test] = {}
        for mode in DM:
            lab_amnd[dn_model][dn_test][mode] = n_omega_calc(
                labels[dn_model][dn_test][mode], multiplicity[dn_test])



### VISUALIZE
print(cb_amnd)
print(nn_amnd)
print(lab_amnd)