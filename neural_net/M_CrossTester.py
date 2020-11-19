#########################################
##### MAIN CROSS-TESTER FOR THE CNN #####
#########################################



from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from M_Model import *
from M_Util import *



## Metadata and hyperparameters
N_CLASSES = 2
REDUCE_DATA = False 
MAKE_SHUFFLE = False
MAKE_NOISE = False
UNSHUFFLE = False
BATCH_SIZE = 32
MODE = 'spec'
BOUND = 3
MODEL_NAME = 'EPOS_EVprof'
TEST_NAME = 'EPOS_Tprof'
ALIGN_NRG = False

if ALIGN_NRG:
    NRG_NAME = '_' + MODEL_NAME
else:
    NRG_NAME = '_' + TEST_NAME

if MODEL_NAME == TEST_NAME:
    REDUCE_DATA = True



if __name__ == '__main__':


    ### Prepare data ###
    
    ## Load data
    dataset_path = '../../DATASETS/NA61_{}/'.format(TEST_NAME)
    dl_test = []  # [features_np_cen, features_np_per, features_np_sum, labels] 
    dl_test.append(np.load(dataset_path + 'features_central.npy'))
    dl_test.append(np.load(dataset_path + 'features_peripheral.npy'))
    dl_test.append(np.load(dataset_path + 'features_sum.npy'))
    if MODE == 'nrg':
        postfix = NRG_NAME
    else:
        postfix = ''
    dl_test.append(np.load(dataset_path + 'labels_{}{}.npy'.format(MODE, postfix)))
    TEST_SIZE = dl_test[0].shape[0]

    ## Unshuffling
    #model_dir = '../../MODELS/{}/{}/{}/'.format(MODEL_NAME, MODE, BOUND)
    model_dir = '../../MODELS/{}/{}/{}/'.format(MODEL_NAME, MODE, BOUND)
    if UNSHUFFLE:
        perm = np.load(model_dir + 'perm0.npy')
        for i in range(len(dl_test)):
            dl_test[i] = dl_test[i][perm]

    ## Reduce data
    if REDUCE_DATA:
        red0 = int(0.75 * TEST_SIZE)
        red1 = TEST_SIZE
        TEST_SIZE = red1 - red0
        for idx, data_agent in enumerate(dl_test):
            dl_test[idx] = data_agent[red0:red1]

    print(TEST_SIZE)

    ## Shuffle data
    if MAKE_SHUFFLE:
        dl_test = shuffleData(dl_test, TEST_SIZE)

    ## Make a noise for the valid dataset
    if MAKE_NOISE:
        dl_test = makeNoise(dl_test, 0, TEST_SIZE, TEST_SIZE, CENTRAL_SHAPE, PERIPHERAL_SHAPE, r_disp=0.5)


    ### Test the model ###

    ## Create a graph 
    model_classifier = M_Model(N_CLASSES)

    ## Define placeholders and make a tf dataset
    with tf.variable_scope('input'):
        plc_cen_feat = tf.placeholder(tf.float32, 
                shape = [None, CENTRAL_SHAPE[0], CENTRAL_SHAPE[1], CENTRAL_SHAPE[2], 1], 
                name='plc_cen_feat')
        plc_per_feat = tf.placeholder(tf.float32, 
                shape = [None, PERIPHERAL_SHAPE[0], PERIPHERAL_SHAPE[1], PERIPHERAL_SHAPE[2], 1], 
                name='plc_per_feat')
        plc_sum_feat = tf.placeholder(tf.float32, 
                shape = [None, 1], 
                name='plc_sum_feat')
        plc_lab = tf.placeholder(tf.float32, 
                shape = [None, N_CLASSES], 
                name='plc_lab')
        plc_batch_size = tf.placeholder(tf.int64, name='plc_batch_size')

    dataset = tf.data.Dataset.from_tensor_slices((plc_cen_feat, plc_per_feat, plc_sum_feat, plc_lab)).batch(plc_batch_size)

    ## Initialize parameters (or loops)
    test_batch_size = TEST_BATCHES[0]   # -//- 
    filters = FILTERS[0]                # -//- 
    kernels = KERNELS[0]                # -//- 
    strides = STRIDES[0]                # -//- 
    paddings = PADDINGS[0]              # -//- 
    fc_dim = FC_DIM[0]                  # -//-

    num_test_batches = TEST_SIZE // test_batch_size 

    ## Make iterators and initiate the model
    dataset_iter = dataset.make_initializable_iterator()
    it_cen, it_per, it_sum, it_lab = dataset_iter.get_next()
    predictions = model_classifier.modelForward(it_cen, it_per, it_sum, filters, kernels, strides, paddings, fc_dim)
    correct_prediction = tf.equal(tf.argmax(it_lab, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    #t_vars = tf.trainable_variables()

    ## For future analysis
    true_lab = tf.argmax(it_lab, 1)
    pred_softmax = tf.nn.softmax(predictions)

    ## Init session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer()) 
        sess.run(tf.local_variables_initializer()) 
        model_weights_save_dir = model_dir + 'MotoNet'
        saver.restore(sess, model_weights_save_dir)
            
        sess.run(dataset_iter.initializer, feed_dict={plc_cen_feat : dl_test[0],
                                                      plc_per_feat : dl_test[1],
                                                      plc_sum_feat : dl_test[2],
                                                      plc_lab : dl_test[3],
                                                      plc_batch_size : test_batch_size})

        test_tot_acc = 0
        test_predictions = []
        for batch_iter in range(num_test_batches):
            acc_val, batch_pred = sess.run([accuracy, pred_softmax], feed_dict={plc_lab : dl_test[3]})
            test_tot_acc += acc_val
            test_predictions.append(batch_pred)
        test_tot_acc /= num_test_batches
            
        print('A test %.4f' % test_tot_acc)
        
        test_predictions_np = np.asarray(test_predictions)
        np.save('../results/cross_test/test_predictions_{}_{}_{}_{}.npy'.format(MODE, MODEL_NAME, TEST_NAME, BOUND), test_predictions_np)
