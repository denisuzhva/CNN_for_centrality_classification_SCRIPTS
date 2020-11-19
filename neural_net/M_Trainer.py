####################################
##### MAIN TRAINER FOR THE CNN #####
####################################



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
REDUCE_DATA = False
MAKE_SHUFFLE = False
MAKE_NOISE = False
REDUCE_FACTOR = 10
L_RATE = 1*1e-5
BATCH_SIZE = 32
MODES = ['spec']
BOUNDS = np.arange(0, 4)
MODEL_NAME = 'EPOS_EV2prof'
N_EPOCHS = 35
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
            dataset_path = f'../../DATASETS/{MODEL_NAME}/'
            dset_features = {}
            dset_features['central'] = np.load(dataset_path + 'features_central.npy')
            dset_features['peripheral'] = np.load(dataset_path + 'features_peripheral.npy')
            ft_keys = dset_features.keys()
            #if bound == 3:
            #    dset_labels = np.load(dataset_path + f'labels_{mode}.npy')
            #else:
            dset_labels = np.load(dataset_path + f'labels_{mode}{bound}.npy')

            ## Balance data (recommended for NA61)
            if BALANCE_DATA:
                dset_features, dset_labels = balanceDatasets(dset_features, dset_labels)

            ## Calculate dataset sizes
            DATASET_SIZE = dset_labels.shape[0]
            TRAIN_SIZE = int(0.75 * DATASET_SIZE)
            VALID_SIZE = DATASET_SIZE - TRAIN_SIZE
            VALID_SIZE = VALID_SIZE // BATCH_SIZE * BATCH_SIZE
            TRAIN_SIZE = DATASET_SIZE - VALID_SIZE

            ## Here one may reduce the datasets
            if REDUCE_DATA:
                dset_features, dset_labels, TRAIN_SIZE, VALID_SIZE, DATASET_SIZE = reduceData(dset_features, dset_labels, TRAIN_SIZE, REDUCE_FACTOR)

            ## Make a noise for the valid dataset
            if MAKE_NOISE:
                dset_features = makeNoise(dset_features, TRAIN_SIZE)


            ### Train and validate ###

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
                plc_is_train = tf.placeholder(tf.bool, name='plc_is_train')
            
            list_plc_features = []
            for key in dict_plc_features.keys():
                list_plc_features.append(dict_plc_features[key])
            tf_dset = tf.data.Dataset.from_tensor_slices((*list_plc_features, plc_lab)).batch(BATCH_SIZE)

            num_train_batches = TRAIN_SIZE // BATCH_SIZE
            num_valid_batches = VALID_SIZE // BATCH_SIZE

            ## Make iterators and initiate the model
            tf_dset_iter = tf_dset.make_initializable_iterator()
            di_next = tf_dset_iter.get_next()
            di_features = di_next[:-1]
            di_labels = di_next[-1]

            #t_vars = tf.trainable_variables()

            ## Loss & optimizer
            predictions = model_classifier.modelForward(*di_features, plc_is_train)
            cross_entropy_logits = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=di_labels, logits=predictions))
            train_opt = tf.train.AdamOptimizer(L_RATE).minimize(cross_entropy_logits)
            correct_prediction = tf.equal(tf.argmax(di_labels, 1), tf.argmax(predictions, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

            ## For future analysis
            pred_softmax = tf.nn.softmax(predictions)

            ## Init session
            for run in range(N_RUNS):
                print("**** RUN %i ****" % run)

                model_dir = f'../../MODELS/{MODEL_NAME}/{mode}/{bound}/{run}/'
                os.makedirs(model_dir, exist_ok=True)
                model_weights_save_file = model_dir + 'MotoNet'

                # separate train and validation data
                if MAKE_SHUFFLE:
                    dset_features_new, dset_labels_new, perm = shuffleData(dset_features, dset_labels)
                    np.save(model_dir + 'perm.npy', perm)
                else:
                    dset_features_new, dset_labels_new = dset_features, dset_labels

                dset_features_train = {}
                dset_features_valid = {}
                for key in ft_keys:
                    dset_features_train[key] = dset_features_new[key][:TRAIN_SIZE]
                    dset_features_valid[key] = dset_features_new[key][TRAIN_SIZE:]
                dset_labels_train = dset_labels_new[:TRAIN_SIZE]
                dset_labels_valid = dset_labels_new[TRAIN_SIZE:]

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    sess.run(tf.global_variables_initializer()) 
                    sess.run(tf.local_variables_initializer()) 
                        
                    valid_acc_max = 0
                    for epoch_iter in range(N_EPOCHS):
                            
                        # train iter init and run
                        train_feed_features = {}
                        for key in ft_keys:                                                                
                            train_feed_features[dict_plc_features[key]] = dset_features_train[key]

                        sess.run(tf_dset_iter.initializer, feed_dict={**train_feed_features, 
                                                                      **{plc_lab : dset_labels_train,
                                                                         plc_is_train : True}})

                        train_tot_loss = 0
                        train_tot_acc = 0
                        for batch_iter in range(num_train_batches):
                            _, loss_val, acc_val = sess.run([train_opt, cross_entropy_logits, accuracy], feed_dict={plc_lab : dset_labels_train,
                                                                                                                    plc_is_train : True})
                            train_tot_loss += loss_val
                            train_tot_acc += acc_val
                        train_tot_loss /= num_train_batches
                        train_tot_acc /= num_train_batches

                        # validation iter init and run
                        valid_feed_features = {}
                        for key in ft_keys:                                                                
                            valid_feed_features[dict_plc_features[key]] = dset_features_valid[key]

                        sess.run(tf_dset_iter.initializer, feed_dict={**valid_feed_features, 
                                                                      **{plc_lab : dset_labels_valid,
                                                                         plc_is_train : False}})

                        valid_tot_loss = 0
                        valid_tot_acc = 0
                        valid_predictions = []
                        for batch_iter in range(num_valid_batches):
                            loss_val, acc_val, batch_predictions = sess.run([cross_entropy_logits, 
                                                                                           accuracy, 
                                                                                           pred_softmax], feed_dict={plc_lab : dset_labels_valid, 
                                                                                                                     plc_is_train : False})
                            valid_tot_loss += loss_val
                            valid_tot_acc += acc_val
                            valid_predictions.append(batch_predictions.tolist())
                        valid_tot_loss /= num_valid_batches
                        valid_tot_acc /= num_valid_batches
                        
                        if valid_tot_acc > valid_acc_max:

                            save_path = saver.save(sess, model_weights_save_file)
                            valid_predictions_np = np.asarray(valid_predictions)
                            np.save(model_dir + 'prediction.npy', valid_predictions_np)
                            valid_acc_max = valid_tot_acc
                            print("epoch %i / L train %.4f / L valid %.4f / A valid max %.4f" % (epoch_iter, 
                                train_tot_loss, 
                                valid_tot_loss, 
                                valid_acc_max))
                            print("Model saved: ", save_path) 
                        else:
                            print("epoch %i" % epoch_iter)

                    sess.close()
                    