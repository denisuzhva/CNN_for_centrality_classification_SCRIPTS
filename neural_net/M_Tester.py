###################################
##### MAIN TESTER FOR THE CNN #####
###################################



from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from M_Models import *
from M_Util import *



## Metadata and hyperparameters
N_CLASSES = 2
BALANCE_DATA = True
MAKE_SHUFFLE = False
UNSHUFFLE = False
BATCH_SIZE = 100
MODES = ['spec']
BOUNDS = np.arange(0, 4)
MODEL_NAME = 'EPOS_EV2prof'
TEST_NAME = 'EPOS_T2prof'
N_RUNS = 5



if __name__ == '__main__':


    ### MODE LOOP ###

    print(tf.__version__)

    for mode in MODES:
        
        print("mode: " + mode)


        ### BOUND LOOP ###

        for bound in BOUNDS:

            print("bound: %i" % bound)
            tf.reset_default_graph()


            ### Prepare data ###
            
            ## Load data
            dataset_path = f'../../DATASETS/{TEST_NAME}/'
            dset_features = {}
            dset_features['central'] = np.load(dataset_path + 'features_central.npy')
            dset_features['peripheral'] = np.load(dataset_path + 'features_peripheral.npy')
            ft_keys = dset_features.keys()
            #if bound == 3:
            #    dset_labels = np.load(dataset_path + f'labels_{mode}.npy')
            #else:
            dset_labels = np.load(dataset_path + f'labels_{mode}{bound}.npy')

            ## Calculate dataset sizes
            TEST_SIZE = dset_labels.shape[0]
            TEST_SIZE = TEST_SIZE // BATCH_SIZE * BATCH_SIZE

           
            ### Test the model ###

            ## Create a graph
            model_classifier = CenPerConv(N_CLASSES)

            ## Define placeholders and make a tf dataset
            with tf.variable_scope('input'):
                dict_plc_features = {}
                for key in ft_keys:
                    ft_tensor_dim = dset_features[key].shape[1:-1]
                    dict_plc_features[key] = tf.placeholder(tf.float32, 
                        shape = [None, ft_tensor_dim[0], ft_tensor_dim[1], ft_tensor_dim[2], 1], 
                        name=f'plc_{key}_features')
                plc_lab = tf.placeholder(tf.float32, 
                    shape = [None, N_CLASSES], 
                    name='plc_lab')

                            
            list_plc_features = []
            for key in dict_plc_features.keys():
                list_plc_features.append(dict_plc_features[key])
            tf_dset = tf.data.Dataset.from_tensor_slices((*list_plc_features, plc_lab)).batch(BATCH_SIZE)

            num_test_batches = TEST_SIZE // BATCH_SIZE

            ## Make iterators and initiate the model
            tf_dset_iter = tf_dset.make_initializable_iterator()
            di_next = tf_dset_iter.get_next()
            di_features = di_next[:-1]
            di_labels = di_next[-1]

            #t_vars = tf.trainable_variables()

            ## Predictions
            predictions = model_classifier.modelForward(*di_features)
            pred_softmax = tf.nn.softmax(predictions)
            pred_argmax = tf.argmax(pred_softmax, axis=1)
            correct_prediction = tf.equal(tf.argmax(di_labels, 1), tf.argmax(predictions, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

            ## Init session
            for run in range(N_RUNS):
                print("**** RUN %i ****" % run)

                model_dir = f'../../MODELS/{MODEL_NAME}/{mode}/{bound}/{run}/'
                test_pred_dir = f'../../TESTS/{MODEL_NAME}__{TEST_NAME}/{mode}/{bound}/{run}/'
                os.makedirs(test_pred_dir, exist_ok=True)
                model_weights_save_file = model_dir + 'MotoNet'

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    sess.run(tf.global_variables_initializer()) 
                    sess.run(tf.local_variables_initializer()) 
                    saver.restore(sess, model_weights_save_file)

                    test_feed_features = {}
                    for key in ft_keys:                                                                
                        test_feed_features[dict_plc_features[key]] = dset_features[key]

                    sess.run(tf_dset_iter.initializer, feed_dict={**test_feed_features, 
                                                                  **{plc_lab : dset_labels}})

                    test_tot_acc = 0
                    test_predictions = []
                    for batch_iter in range(num_test_batches):
                        acc_val, batch_pred = sess.run([accuracy, pred_argmax], feed_dict={plc_lab : dset_labels})
                        test_tot_acc += acc_val
                        test_predictions.append(batch_pred)
                    test_tot_acc /= num_test_batches

                    print('A test %.4f' % test_tot_acc)
                    test_predictions_np = np.asarray(test_predictions).flatten()
                    np.save(test_pred_dir + 'test_predictions.npy', test_predictions_np)

                    sess.close()
                    