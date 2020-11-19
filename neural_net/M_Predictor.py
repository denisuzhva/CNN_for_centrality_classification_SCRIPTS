######################################
##### MAIN PREDICTOR FOR THE CNN #####
######################################



from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from M_Model import *
from M_Util import *



## Metadata and hyperparameters
N_CLASSES = 2
CENTRAL_SHAPE = [4, 4, 10]      # W H D
PERIPHERAL_SHAPE = [4, 4, 10]   # W H D
TEST_SIZE = 0
REDUCE_DATA = False 
MAKE_SHUFFLE = False
MAKE_NOISE = False
TEST_BATCHES = [100]
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
MODE = "spec"
BOUND = 0
MODEL_NAME = "EPOSprof"
TEST_NAME = "BeBeExperProf"

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
    TEST_SIZE = dl_test[0].shape[0]
    print(TEST_SIZE)

    if REDUCE_DATA:
        red0 = int(0.75 * TEST_SIZE)
        red1 = TEST_SIZE
        TEST_SIZE = red1 - red0
        for idx, data_agent in enumerate(dl_test):
            dl_test[idx] = data_agent[red0:red1]

    ## Shuffle data
    if MAKE_SHUFFLE:
        dl_test, perm = shuffleData(dl_test, TEST_SIZE)

    ## Make a noise for the valid dataset
    if MAKE_NOISE:
        dl_test = makeNoise(dl_test, 0, TEST_SIZE, TEST_SIZE, CENTRAL_SHAPE, PERIPHERAL_SHAPE, r_disp=0.5)

    ## Model directory
    model_weights_save_dir = "../../MODELS/{}/{}/{}/MotoNet".format(MODEL_NAME, MODE, BOUND)


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
        plc_batch_size = tf.placeholder(tf.int64, name='plc_batch_size')

    dataset = tf.data.Dataset.from_tensor_slices((plc_cen_feat, plc_per_feat, plc_sum_feat)).batch(plc_batch_size)

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
    it_cen, it_per, it_sum = dataset_iter.get_next()
    predictions = model_classifier.modelForward(it_cen, it_per, it_sum, filters, kernels, strides, paddings, fc_dim)
    pred_softmax = tf.nn.softmax(predictions)
    pred_label = tf.argmax(pred_softmax, 1)

    #t_vars = tf.trainable_variables()

    ## Init session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer()) 
        sess.run(tf.local_variables_initializer()) 
        saver.restore(sess, model_weights_save_dir)
            
        sess.run(dataset_iter.initializer, feed_dict={plc_cen_feat : dl_test[0],
                                                      plc_per_feat : dl_test[1],
                                                      plc_sum_feat : dl_test[2],
                                                      plc_batch_size : test_batch_size})

        test_tot_acc = 0
        test_predictions = []
        for batch_iter in range(num_test_batches):
            batch_pred = sess.run([pred_label], feed_dict={})
            test_predictions.append(batch_pred)
        test_tot_acc /= num_test_batches
            
        print("A test %.4f" % test_tot_acc)
        
        test_predictions_np = np.asarray(test_predictions)
        np.save('../results/cross_test/test_predictions_{}_{}_{}.npy'.format(MODE, MODEL_NAME, TEST_NAME), test_predictions_np)
