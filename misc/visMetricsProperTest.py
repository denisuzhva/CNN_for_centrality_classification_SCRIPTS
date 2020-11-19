import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



## Metadata

SPEC_BOUNDS = np.arange(0, 4)
RUNS = 5
DATASET = 'EPOS_T2prof'
TRAIN_DATASET = 'EPOS_EV2prof'
DATA_DIR = f'../../DATASETS/{DATASET}/'
MODEL_DIR = f'../../MODELS/{TRAIN_DATASET}/'
TEST_DIR = f'../../TESTS/{TRAIN_DATASET}__{DATASET}/'
NRG_STR = 'nrg'
SPEC_STR = 'spec'
MODES = [SPEC_STR]
EMEAS_STR = 'emeas'
PERC_ARR = []


## Metric arrays

# accuracy
for mode in MODES:
    nn_ac_m = {mode: {}}
    nn_ac_e = {mode: {}}
    cb_ac_m = {mode: {}}
    cb_ac_e = {mode: {}}

# average multiplicity
for mode in MODES:
    nn_am_m = {mode: {}}
    nn_am_e = {mode: {}}
cb_am_m = {}
cb_am_e = {}

# scaled variance
for mode in MODES:
    nn_sv_m = {mode: {}}
    nn_sv_e = {mode: {}}
cb_sv_m = {}
cb_sv_e = {}

# labels
for mode in MODES:
    lab_am_m = {mode: {}}
    lab_am_e = {mode: {}}
    lab_sv_m = {mode: {}}
    lab_sv_e = {mode: {}}


## Utility functions

# calculate <N> and omega
def n_omega_calc(lab, mul, do_round=False):
    mul_first_class = mul[lab == 0]
    am = mul_first_class.mean()
    sv = np.power(mul_first_class.std(), 2) / am
    if do_round:
        am = np.round(am, 2)
        sv = np.round(sv, 2)
    return am, sv

# calculate CB accuracy
def eval_cb_ac(x_dat, y_dat, x_bnd, y_bnd):
    dat_size = x_dat.size
    ac_val =  np.sum(np.logical_or(np.logical_and((x_dat <= x_bnd), (y_dat <= y_bnd)), \
            np.logical_and((x_dat > x_bnd), (y_dat > y_bnd)))) / dat_size
    return ac_val


## Process data and resutls

# load data for cut-based analysis
nse_data = {NRG_STR: np.load(DATA_DIR + 'nrg24.npy').flatten(), \
    SPEC_STR: np.load(DATA_DIR + 'spec24.npy').flatten(), \
    EMEAS_STR: np.load(DATA_DIR +  'features_sum.npy').flatten()}
emeas_sorted = np.sort(nse_data[EMEAS_STR])
nrg_sorted = np.sort(nse_data[NRG_STR])

# load multiplicities for CB and labels
mul = np.load(DATA_DIR + 'mul.npy')
n_events_glob = mul.size
   
# iterate over the spec_bounds
for spec_bound in SPEC_BOUNDS:

    # calculate forward and measured energy spec_bounds for CB
    perc = nse_data[SPEC_STR][nse_data[SPEC_STR] <= spec_bound].size / nse_data[SPEC_STR].size
    PERC_ARR.append(perc)
    nse_bounds = {NRG_STR: nrg_sorted[int(perc * nse_data[NRG_STR].size)], \
        SPEC_STR: spec_bound, \
        EMEAS_STR: emeas_sorted[int(perc * nse_data[EMEAS_STR].size)]}

    # iterate over the modes
    for mode in MODES:
        
        # load labels 
        labels = np.load(DATA_DIR + 'labels_{}{}.npy'.format(mode, spec_bound)).argmax(axis=1)
                
        # load and process runs
        ac = np.zeros(RUNS)
        am = np.zeros_like(ac)
        sv = np.zeros_like(ac)
        for run in range(RUNS):

            # load predictions
            predictions = np.load(TEST_DIR + f'{mode}/{spec_bound}/{run}/' + 'test_predictions.npy')
            
            # calculate CNN accuracy
            ac[run] = np.sum(labels == predictions) / n_events_glob

            # calculate CNN am and sv
            am[run], sv[run] = n_omega_calc(predictions, mul)

        # calculate CNN accuracy, am and sv (mean and std)
        nn_ac_m[mode][spec_bound] = np.mean(ac)
        nn_ac_e[mode][spec_bound] = np.std(ac, ddof=1)
        nn_am_m[mode][spec_bound] = np.mean(am)
        nn_am_e[mode][spec_bound] = np.std(am, ddof=1)
        nn_sv_m[mode][spec_bound] = np.mean(sv)
        nn_sv_e[mode][spec_bound] = np.std(sv, ddof=1)

        # calculate CB accuracy
        cb_ac_m[mode][spec_bound] = eval_cb_ac(nse_data[mode], nse_data[EMEAS_STR], \
            nse_bounds[mode], nse_bounds[EMEAS_STR])
        cb_ac_sub = eval_cb_ac(nse_data[mode][:n_events_glob//10], nse_data[EMEAS_STR][:n_events_glob//10], \
            nse_bounds[mode], nse_bounds[EMEAS_STR])
        cb_ac_e[mode][spec_bound] = np.abs(cb_ac_m[mode][spec_bound] - cb_ac_sub)

        # calculate label am and sv
        lab_am_m[mode][spec_bound], lab_sv_m[mode][spec_bound] = n_omega_calc(labels, mul)
        lab_am_sub, lab_sv_sub = n_omega_calc(labels[:labels.size//10], mul[:mul.size//10])
        lab_am_e[mode][spec_bound] = np.abs(lab_am_m[mode][spec_bound] - lab_am_sub)
        lab_sv_e[mode][spec_bound] = np.abs(lab_sv_m[mode][spec_bound] - lab_sv_sub)

    # calculate CB am and sv
    pred_cb = np.zeros((n_events_glob), dtype=np.uint8)
    pred_cb[nse_data[EMEAS_STR] >= nse_bounds[EMEAS_STR]] = 1

    cb_am_m[spec_bound], cb_sv_m[spec_bound] = n_omega_calc(pred_cb, mul) 
    cb_am_sub, cb_sv_sub = n_omega_calc(pred_cb[:pred_cb.size//10], mul[:mul.size//10])
    cb_am_e[spec_bound] = np.abs(cb_am_m[spec_bound] - cb_am_sub)
    cb_sv_e[spec_bound] = np.abs(cb_sv_m[spec_bound] - cb_sv_sub)


## Preprocess the results

perc_arr_trun = np.round(np.array(PERC_ARR) * 100, 2) 
modes_better = {NRG_STR: 'fwd energy', SPEC_STR: 'num. spec.'}


## Plot results

# accuracies
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_ac_m[mode].values(), yerr=nn_ac_e[mode].values(), label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_ac_m[mode].values(), yerr=cb_ac_e[mode].values(), label='CBA', color='r')
    ax.legend(loc='lower left')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Accuracy')
    ax.set_title('CNN against CBA ({})'.format(modes_better[mode]))
    fig.savefig(MODEL_DIR + 'ac_{}.png'.format(mode))

# <N>
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_am_m[mode].values(), yerr=nn_am_e[mode].values(), label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_am_m.values(), yerr=cb_am_e.values(), label='CBA', color='r')
    ax.errorbar(perc_arr_trun, lab_am_m[mode].values(), yerr=lab_am_e[mode].values(), label='label', color='g')
    ax.legend(loc='upper right')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Average multiplicity')
    ax.set_title('CNN against CBA and labels ({})'.format(modes_better[mode]))
    fig.savefig(MODEL_DIR + 'am_{}.png'.format(mode))

# SV
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_sv_m[mode].values(), yerr=nn_sv_e[mode].values(), label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_sv_m.values(), yerr=cb_sv_e.values(), label='CBA', color='r')
    ax.errorbar(perc_arr_trun, lab_sv_m[mode].values(), yerr=lab_sv_e[mode].values(), label='label', color='g')
    ax.legend(loc='lower right')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Scaled variance')
    ax.set_title(' CNN against CBA and labels ({})'.format(modes_better[mode]))
    fig.savefig(MODEL_DIR + 'sv_{}.png'.format(mode))

# pyplot misc
plt.tight_layout()
plt.show()

