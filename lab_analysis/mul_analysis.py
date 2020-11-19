import numpy as np



## Metadata
DATASET_SIZE = 80000
TEST_SIZE = 20000
TRAIN_SIZE = DATASET_SIZE - TEST_SIZE


## Load data
all_mul = np.load('../../DATASETS/NA61_cen_per_separated/all/mul.npy')

pred_nn_nrg = np.load('./test_predictions_nrg.npy')
pred_nn_spec = np.load('./test_predictions_spec.npy')

cen_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_peripheral.npy')

lab_nrg = np.load('../../DATASETS/NA61_cen_per_separated/all/labels_nrg.npy')
lab_spec = np.load('../../DATASETS/NA61_cen_per_separated/all/labels_spec.npy')


## Preprocess
all_mul = all_mul[:DATASET_SIZE]
lab_nrg  = lab_nrg[:DATASET_SIZE]
lab_spec = lab_spec[:DATASET_SIZE]
lab_nrg = lab_nrg.argmax(axis=1).astype(np.uint8)
lab_spec = lab_spec.argmax(axis=1).astype(np.uint8)
pred_nn_nrg = pred_nn_nrg.reshape(-1, 2).argmax(axis=1)
pred_nn_spec = pred_nn_spec.reshape(-1, 2).argmax(axis=1)


## Make noise
r_disp = 0.2
r_mean = 1.

cen_noise = np.ones_like(cen_modules)
per_noise = np.ones_like(per_modules)
cen_noise_single = np.random.randn(1, 4, 4, 10, 1) * np.sqrt(r_disp) + r_mean
per_noise_single = np.random.randn(1, 4, 4, 10, 1) * np.sqrt(r_disp) + r_mean
cen_noise_act = np.repeat(cen_noise_single, DATASET_SIZE, axis=0)
per_noise_act = np.repeat(per_noise_single, DATASET_SIZE, axis=0)

cen_mod_noised = np.multiply(cen_modules, cen_noise_act)
per_mod_noised = np.multiply(per_modules, per_noise_act)
#cen_mod_noised = cen_modules
#per_mod_noised = per_modules
e_meas = np.sum(cen_mod_noised, (1, 2, 3, 4)).flatten() + np.sum(per_mod_noised, (1, 2, 3, 4)).flatten()


## Find class bounds
e_meas_sorted = np.sort(e_meas)
bound_perc = 15.8
e_meas_bound = e_meas_sorted[int(bound_perc*0.01*DATASET_SIZE)]


## Find CB predictions
pred_cb = np.zeros_like(e_meas, dtype=np.uint8)
pred_cb[e_meas >= e_meas_bound] = 1


## Calculate the <N>s and omegas
mul_lab_nrg = all_mul[lab_nrg == 0]
mul_lab_spec = all_mul[lab_spec == 0]
mul_nn_nrg = all_mul[TRAIN_SIZE:][pred_nn_nrg == 0]
mul_nn_spec = all_mul[TRAIN_SIZE:][pred_nn_spec == 0]
mul_cb = all_mul[pred_cb == 0]

am_lab_nrg = mul_lab_nrg.mean()
am_lab_spec = mul_lab_spec.mean()
am_nn_nrg = mul_nn_nrg.mean()
am_nn_spec = mul_nn_spec.mean()
am_cb = mul_cb.mean()

ndisp_lab_nrg = np.power(mul_lab_nrg.std(), 2) / am_lab_nrg
ndisp_lab_spec = np.power(mul_lab_spec.std(), 2) / am_lab_spec
ndisp_nn_nrg = np.power(mul_nn_nrg.std(), 2) / am_nn_nrg
ndisp_nn_spec = np.power(mul_nn_spec.std(), 2) / am_nn_spec
ndisp_cb = np.power(mul_cb.std(), 2) / am_cb


## Print the results
print('Average multiplicity.')
print('<N> label (E_true): %.2f' % am_lab_nrg)
print('<N> label (n_spect): %.2f' % am_lab_spec)
print('<N> CNN (E_true): %.2f' % am_nn_nrg)
print('<N> CNN (n_spect): %.2f' % am_nn_spec)
print('<N> Cut-based: %.2f' % am_cb)

print('Scaled variance.')
print('Scaled variane (omega) label (E_true): %.2f' % ndisp_lab_nrg)
print('Scaled variane (omega) label (n_spect): %.2f' % ndisp_lab_spec)
print('Scaled variane (omega) CNN (E_true): %.2f' % ndisp_nn_nrg)
print('Scaled variane (omega) CNN (n_spect): %.2f' % ndisp_nn_spec)
print('Scaled variane (omega) Cut-based: %.2f' % ndisp_cb)
