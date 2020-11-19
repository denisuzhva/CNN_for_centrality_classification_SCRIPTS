from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import numpy as np
from M_Model import *
from M_Util import *
from M_Tester import *



## Metadata and parameters
DATASET_SIZE = 80000
TEST_SIZE = 20000
TRAIN_SIZE = DATASET_SIZE - TEST_SIZE
r_disp = 0.5
r_mean = 1.
n_checks = 20
N_CLASSES = 2
CENTRAL_SHAPE = [4, 4, 10]      # W H D
PERIPHERAL_SHAPE = [4, 4, 10]   # W H D
REDUCE_DATA = False
MAKE_SHUFFLE = False
MAKE_NOISE = True
REDUCED_SIZE = 8000
BATCHES = [100]
FILTERS = [                             # global -> layer
            [128, 128]
          ]
FC_DIM = [                              # global -> layer
            [1024]
         ]
KERNELS = [                             # global -> layer -> dimension
            [[3, 3, 1], [3, 3, 6]]
          ]      
STRIDES = [                             # global -> layer -> dimension
            [[1, 1, 1], [1, 1, 2]]
          ]
PADDINGS = [                            # global -> layer
            ["VALID", "VALID"]
           ]
MODES = ["nrg", "spec"]
MODEL_NAME = "M_1"


## Load data
all_mul = np.load('../../DATASETS/NA61_cen_per_separated/all/mul.npy')

cen_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_central.npy')
per_modules = np.load('../../DATASETS/NA61_cen_per_separated/all/features_peripheral.npy')

lab_nrg = np.load('../../DATASETS/NA61_cen_per_separated/all/labels_nrg.npy')
lab_spec = np.load('../../DATASETS/NA61_cen_per_separated/all/labels_spec.npy')

lab_nrg = lab_nrg.argmax(axis=1).astype(np.uint8)
lab_spec = lab_spec.argmax(axis=1).astype(np.uint8)


## Define placeholders and make a tf dataset
with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
    plc_cen_feat = tf.placeholder(tf.float32, 
            shape = [None, CENTRAL_SHAPE[0], CENTRAL_SHAPE[1], CENTRAL_SHAPE[2], 1], 
            name='plc_cen_feat')
    plc_per_feat = tf.placeholder(tf.float32, 
            shape = [None, PERIPHERAL_SHAPE[0], PERIPHERAL_SHAPE[1], PERIPHERAL_SHAPE[2], 1], 
            name='plc_per_feat')
    plc_sum_feat = tf.placeholder(tf.float32, 
            shape = [None, 1], 
            name='plc_sum_feat')
    plc_batch_size = tf.placeholder(tf.int64, name='plc_batch_size')

dataset = tf.data.Dataset.from_tensor_slices((plc_cen_feat, plc_per_feat, plc_sum_feat)).batch(plc_batch_size)


## Initialize parameters (or loops)
batch_size = BATCHES[0] # possible to make a loop 
filters = FILTERS[0]    # -//- 
kernels = KERNELS[0]    # -//- 
strides = STRIDES[0]    # -//- 
paddings = PADDINGS[0]  # -//- 
fc_dim = FC_DIM[0]      # -//-

num_batches = TEST_SIZE // batch_size


## Make iterators and initiate the model
model_classifier = M_Model(N_CLASSES)
dataset_iter = dataset.make_initializable_iterator()
it_cen, it_per, it_sum = dataset_iter.get_next()
predictions = model_classifier.modelForward(it_cen, it_per, it_sum, filters, kernels, strides, paddings, fc_dim, 1.0, False)
pred_lab = tf.argmax(predictions, 1)


## Make tensors to store results
am_nn_mat = np.zeros((2, n_checks))
ndisp_nn_mat = np.zeros((2, n_checks))
am_cb_mat = np.zeros((n_checks))
ndisp_cb_mat = np.zeros((n_checks))


## Init session
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer()) 

    for mode_count, _ in enumerate(MODES):
        model_weights_save_dir = "../../MODELS/{0}/{1}/MotoNet".format(MODEL_NAME, MODES[mode_count])
        saver.restore(sess, model_weights_save_dir)

        for ittt in range(n_checks):
        
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

            sess.run(dataset_iter.initializer, feed_dict={plc_cen_feat : cen_mod_noised[TRAIN_SIZE:, :, :, :, :],
                                                            plc_per_feat : per_mod_noised[TRAIN_SIZE:, :, :, :, :],
                                                            plc_sum_feat : e_meas[TRAIN_SIZE:].reshape((-1, 1)),
                                                            plc_batch_size : batch_size})
            test_predictions = np.array([])
            for batch_iter in range(num_batches):
                batch_pred_label = sess.run([pred_lab], feed_dict={})
                test_predictions = np.append(test_predictions, batch_pred_label[0])

            mul_nn = all_mul[TRAIN_SIZE:][test_predictions == 0]
            am_nn = mul_nn.mean()
            ndisp_nn = np.power(mul_nn.std(), 2) / am_nn
            
            am_nn_mat[mode_count, ittt] = am_nn
            ndisp_nn_mat[mode_count, ittt] = ndisp_nn 

            if mode_count == 0:
                e_meas_sorted = np.sort(e_meas)
                bound_perc = 15.8
                e_meas_bound = e_meas_sorted[int(bound_perc*0.01*DATASET_SIZE)]

                pred_cb = np.zeros_like(e_meas, dtype=np.uint8)
                pred_cb[e_meas >= e_meas_bound] = 1

                mul_cb = all_mul[pred_cb == 0]
                am_cb = mul_cb.mean()
                ndisp_cb = np.power(mul_cb.std(), 2) / am_cb

                am_cb_mat[ittt] = am_cb
                ndisp_cb_mat[ittt] = ndisp_cb


## Calculate statistics
am_nn_nrg_av = np.mean(am_nn_mat[0, :])
am_nn_nrg_sd = np.std(am_nn_mat[0, :])
am_nn_spec_av = np.mean(am_nn_mat[1, :])
am_nn_spec_sd = np.std(am_nn_mat[1, :])
am_cb_av = np.mean(am_cb_mat)
am_cb_sd = np.std(am_cb_mat)

ndisp_nn_nrg_av = np.mean(ndisp_nn_mat[0, :])
ndisp_nn_nrg_sd = np.std(ndisp_nn_mat[0, :])
ndisp_nn_spec_av = np.mean(ndisp_nn_mat[1, :])
ndisp_nn_spec_sd = np.std(ndisp_nn_mat[1, :])
ndisp_cb_av = np.mean(ndisp_cb_mat)
ndisp_cb_sd = np.std(ndisp_cb_mat)


## Calculate labels
mul_lab_nrg = all_mul[lab_nrg == 0]
mul_lab_spec = all_mul[lab_spec == 0]
am_lab_nrg = mul_lab_nrg.mean()
am_lab_spec = mul_lab_spec.mean()
ndisp_lab_nrg = np.power(mul_lab_nrg.std(), 2) / am_lab_nrg
ndisp_lab_spec = np.power(mul_lab_spec.std(), 2) / am_lab_spec


## Display the results
print('Dispersion: %.2f' % r_disp)
print('Average multiplicity.')
print('<N> label (E_true): %.2f' % am_lab_nrg)
print('<N> label (n_spect): %.2f' % am_lab_spec)
print('<N> CNN (E_true): %.2f +- %.2f' % (am_nn_nrg_av, am_nn_nrg_sd))
print('<N> CNN (n_spect): %.2f +- %.2f' % (am_nn_spec_av, am_nn_spec_sd))
print('<N> Cut-based: %.2f +- %.2f' % (am_cb_av, am_cb_sd))

print('Scaled variance.')
print('Scaled variane (omega) label (E_true): %.2f' % ndisp_lab_nrg)
print('Scaled variane (omega) label (n_spect): %.2f' % ndisp_lab_spec)
print('Scaled variane (omega) CNN (E_true): %.2f +- %.2f' % (ndisp_nn_nrg_av, ndisp_nn_nrg_sd))
print('Scaled variane (omega) CNN (n_spect): %.2f +- %.2f' % (ndisp_nn_spec_av, ndisp_nn_spec_sd))
print('Scaled variane (omega) Cut-based: %.2f +- %.2f' % (ndisp_cb_av, ndisp_cb_sd))


fig, axs = plt.subplots(2, 2, figsize=(15, 15), dpi=120)
axs[0, 0].hist(am_nn_mat[0, :])
axs[0, 1].hist(ndisp_nn_mat[0, :])
axs[1, 0].hist(am_cb_mat)
axs[1, 1].hist(ndisp_cb_mat)
#plt.yscale('log')
fig.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
fig.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.show()