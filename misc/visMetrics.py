import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'



## Metadata

MODES = ['spec']
SPEC_BOUNDS = 7
RUNS = 5
MAKE_SHUFFLE = False
DATASET = 'EPOS_400prof'
DATA_DIR = '../../DATASETS/NA61_{}/'.format(DATASET)
RESULTS_DIR = '../../MODELS/{}/'.format(DATASET)
NRG_STR = 'nrg'
SPEC_STR = 'spec'
EMEAS_STR = 'emeas'
PERC_ARR = np.zeros(SPEC_BOUNDS)


## Metric arrays

# accuracy
for mode in MODES:
    nn_ac_m = {mode: np.zeros(SPEC_BOUNDS)}
    nn_ac_e = {mode: np.zeros(SPEC_BOUNDS)}
    cb_ac_m = {mode: np.zeros(SPEC_BOUNDS)}
    cb_ac_e = {mode: np.zeros(SPEC_BOUNDS)}

# average multiplicity
for mode in MODES:
    nn_am_m = {mode: np.zeros(SPEC_BOUNDS)}
    nn_am_e = {mode: np.zeros(SPEC_BOUNDS)}
cb_am_m = np.zeros(SPEC_BOUNDS)
cb_am_e = np.zeros(SPEC_BOUNDS)

# scaled variance
for mode in MODES:
    nn_sv_m = {mode: np.zeros(SPEC_BOUNDS)}
    nn_sv_e = {mode: np.zeros(SPEC_BOUNDS)}
cb_sv_m = np.zeros(SPEC_BOUNDS)
cb_sv_e = np.zeros(SPEC_BOUNDS)

# labels
for mode in MODES:
    lab_am_m = {mode: np.zeros(SPEC_BOUNDS)}
    lab_am_e = {mode: np.zeros(SPEC_BOUNDS)}
    lab_sv_m = {mode: np.zeros(SPEC_BOUNDS)}
    lab_sv_e = {mode: np.zeros(SPEC_BOUNDS)}


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

# load multiplicities
mul = np.load(DATA_DIR + 'mul.npy')
N_EVENTS = mul.size
train_size = int(0.75 * N_EVENTS)
finish_point = N_EVENTS - 50
   
# iterate over the spec_bounds
for spec_bound in range(SPEC_BOUNDS):

    # calculate forward and measured energy spec_bounds for CB
    perc = nse_data[SPEC_STR][nse_data[SPEC_STR] <= spec_bound].size / nse_data[SPEC_STR].size
    PERC_ARR[spec_bound] = perc
    nse_bounds = {NRG_STR: nrg_sorted[int(perc * nse_data[NRG_STR].size)], \
        SPEC_STR: spec_bound, \
        EMEAS_STR: emeas_sorted[int(perc * nse_data[EMEAS_STR].size)]}

    # iterate over the modes
    for mode in MODES:
        
        # load labels
        labels = np.load(DATA_DIR + 'labels_{}{}.npy'.format(mode, spec_bound)).argmax(axis=1)
        labels_rdy = copy(labels)
        mul_rdy = copy(mul)
        
        # load and process runs
        b_dir = RESULTS_DIR + '{}/{}/'.format(mode, spec_bound)
        ac = np.zeros(RUNS)
        am = np.zeros_like(ac)
        sv = np.zeros_like(ac)
        for run in range(RUNS):
            
            # load permutations and predictions
            if MAKE_SHUFFLE:
                perm = np.load(b_dir + 'perm{}.npy'.format(run))
            predictions = np.load(b_dir + 'prediction{}.npy'.format(run)).reshape((-1, 2)).argmax(axis=1)

            # permute labels and multiplicities 
            if MAKE_SHUFFLE:
                labels_rdy = labels_rdy[perm]
                mul_rdy = mul_rdy[perm]
            
            # calculate accuracy
            ac[run] = np.sum(labels_rdy[train_size:finish_point] == predictions) / (finish_point - train_size)

            # calculate average multiplicity and scaled variance
            am[run], sv[run] = n_omega_calc(predictions, mul_rdy[train_size:finish_point])

        # calculate averages and errors (for the CNN)
        nn_ac_m[mode][spec_bound] = np.mean(ac)
        nn_ac_e[mode][spec_bound] = np.std(ac, ddof=1)
        nn_am_m[mode][spec_bound] = np.mean(am)
        nn_am_e[mode][spec_bound] = np.std(am, ddof=1)
        nn_sv_m[mode][spec_bound] = np.mean(sv)
        nn_sv_e[mode][spec_bound] = np.std(sv, ddof=1)

        # calculate CB accuracy
        cb_ac_m[mode][spec_bound] = eval_cb_ac(nse_data[mode], nse_data[EMEAS_STR], \
            nse_bounds[mode], nse_bounds[EMEAS_STR])
        cb_ac_sub = eval_cb_ac(nse_data[mode][:N_EVENTS//10], nse_data[EMEAS_STR][:N_EVENTS//10], \
            nse_bounds[mode], nse_bounds[EMEAS_STR])
        cb_ac_e[mode][spec_bound] = np.abs(cb_ac_m[mode][spec_bound] - cb_ac_sub)

        # calculate label multiplicities and scaled variances
        lab_am_m[mode][spec_bound], lab_sv_m[mode][spec_bound] = n_omega_calc(labels, mul)
        lab_am_sub, lab_sv_sub = n_omega_calc(labels[:labels.size//10], mul[:mul.size//10])
        lab_am_e[mode][spec_bound] = np.abs(lab_am_m[mode][spec_bound] - lab_am_sub)
        lab_sv_e[mode][spec_bound] = np.abs(lab_sv_m[mode][spec_bound] - lab_sv_sub)

    # calculate CB am and sv
    pred_cb = np.zeros((N_EVENTS), dtype=np.uint8)
    pred_cb[nse_data[EMEAS_STR] >= nse_bounds[EMEAS_STR]] = 1

    cb_am_m[spec_bound], cb_sv_m[spec_bound] = n_omega_calc(pred_cb, mul) 
    cb_am_sub, cb_sv_sub = n_omega_calc(pred_cb[:pred_cb.size//10], mul[:mul.size//10])
    cb_am_e[spec_bound] = np.abs(cb_am_m[spec_bound] - cb_am_sub)
    cb_sv_e[spec_bound] = np.abs(cb_sv_m[spec_bound] - cb_sv_sub)


## Preprocess the results

perc_arr_trun = np.round(PERC_ARR * 100, 2) 
modes_better = {NRG_STR: 'fwd energy', SPEC_STR: 'num. spec.'}


## Plot results

# accuracies
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_ac_m[mode], yerr=nn_ac_e[mode], label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_ac_m[mode], yerr=cb_ac_e[mode], label='CBA', color='r')
    ax.legend(loc='lower left')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Accuracy')
    ax.set_title('CNN against CBA ({})'.format(modes_better[mode]))
    fig.savefig(RESULTS_DIR + 'ac_{}.png'.format(mode))

# <N>
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_am_m[mode], yerr=nn_am_e[mode], label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_am_m, yerr=cb_am_e, label='CBA', color='r')
    ax.errorbar(perc_arr_trun, lab_am_m[mode], yerr=lab_am_e[mode], label='label', color='g')
    ax.legend(loc='upper right')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Average multiplicity')
    ax.set_title('CNN against CBA and labels ({})'.format(modes_better[mode]))
    fig.savefig(RESULTS_DIR + 'am_{}.png'.format(mode))

# SV
for mode in MODES:
    fig, ax = plt.subplots()
    ax.errorbar(perc_arr_trun, nn_sv_m[mode], yerr=nn_sv_e[mode], label='CNN', color='b')
    ax.errorbar(perc_arr_trun, cb_sv_m, yerr=cb_sv_e, label='CBA', color='r')
    ax.errorbar(perc_arr_trun, lab_sv_m[mode], yerr=lab_sv_e[mode], label='label', color='g')
    ax.legend(loc='lower right')
    ax.set_xlabel('Centrality, 0-X%')
    ax.set_ylabel('Scaled variance')
    ax.set_title(' CNN against CBA and labels ({})'.format(modes_better[mode]))
    fig.savefig(RESULTS_DIR + 'sv_{}.png'.format(mode))

# pyplot misc
plt.tight_layout()
plt.show()

