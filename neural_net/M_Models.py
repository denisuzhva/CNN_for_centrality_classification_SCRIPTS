#################################
##### THE MODELS OF THE CNN #####
#################################



from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Dense, Flatten, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import abc



## Abstract class for an abstract model
class Model(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def __init__(self,  n_classes):
        self._n_classes = n_classes


    @abc.abstractmethod
    def modelForward(self, in_data, is_train):
        pass


## Central-peripheral CNN
class CenPerConv(Model):


    def __init__(self, n_classes):
        super().__init__(n_classes)
        self.__filters = [32, 64]
        self.__kernels = [[3, 3, 1], [3, 3, 6]]
        self.__strides = [[1, 1, 1], [1, 1, 2]]
        self.__paddings = ['VALID', 'VALID']
        self.__fc_dim = [1024]
        self.__do_prob = 0.2


    def modelForward(self, in_cen, in_per, is_train=True):

        #def probTrain():
        #    return self.__do_prob
        #def probValid():
        #    return 1.0
        #do_prob = tf.cond(tf.equal(is_train, tf.constant(True, dtype=tf.bool)), probTrain, probValid)
        
        with tf.variable_scope('model_scope', reuse=tf.AUTO_REUSE) as scope:

            ## 1 conv -> batch norm -> activate
            conv1 = Conv3D(self.__filters[0], kernel_size=self.__kernels[0], strides=self.__strides[0], padding=self.__paddings[0])(in_cen)
            conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,  updates_collections=None)
            #conv1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv1, training=is_train)
            conv1 = LeakyReLU(alpha=0.2)(conv1)

            ## pad act1 and then add it to in_per (to the peripheral modules)
            cen_paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]) # pad only the cross-section
            conv1_padded = tf.pad(conv1, cen_paddings)
            conv1_combined = tf.add(conv1_padded, in_per)

            ## 2 conv -> batch norm -> activate
            conv2 = Conv3D(self.__filters[1], kernel_size=self.__kernels[1], strides=self.__strides[1], padding=self.__paddings[1])(conv1_combined)
            conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay=0.9,  updates_collections=None)
            #conv2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv2, training=is_train)
            conv2 = LeakyReLU(alpha=0.2)(conv2)

            ## 1 FC 
            fc1 = Flatten()(conv2)
            fc1 = Dense(self.__fc_dim[0],
                kernel_regularizer=l2())(fc1)
            fc1 = LeakyReLU(alpha=0.3)(fc1)
            
            ## Dropout
            fc1_drop = Dropout(self.__do_prob)(fc1, training=is_train)

            ## 2 FC
            logits = Dense(self._n_classes,
                kernel_regularizer=l2())(fc1_drop)

            return logits


## Super-resolution 1
class SuperResCNN(Model):


    def __init__(self, n_classes):
        super().__init__(n_classes)
        self.__filters = [128, 128]
        self.__kernels = [[3, 3, 1], [3, 3, 6]]
        self.__strides = [[1, 1, 1], [1, 1, 2]]
        self.__paddings = ['VALID', 'VALID']
        self.__fc_dim = [1024]
        self.__do_prob = 0.2


    def modelForward(self, in_data, is_train=False):

        in_cen, in_per = in_data
        def probTrain():
            return self.__do_prob
        def probValid():
            return 1.0
        do_prob = tf.cond(tf.equal(is_train, tf.constant(True, dtype=tf.bool)), probTrain, probValid)
        
        with tf.variable_scope('model_scope', reuse=tf.AUTO_REUSE) as scope:

            ## 1 conv -> batch norm -> activate
            conv1 = Conv3D(self.__filters[0], kernel_size=self.__kernels[0], strides=self.__strides[0], padding=self.__paddings[0])(in_cen)
            conv1 = BatchNormalization(momentum=0.9)(conv1, is_train)
            conv1 = LeakyReLU()(conv1)

            ## pad act1 and then add it to in_per (to the peripheral modules)
            cen_paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]) # pad only the cross-section
            conv1_padded = tf.pad(conv1, cen_paddings)
            conv1_combined = tf.add(conv1_padded, in_per)

            ## 2 conv -> batch norm -> activate
            conv2 = Conv3D(self.__filters[1], kernel_size=self.__kernels[1], strides=self.__strides[1], padding=self.__paddings[1])(conv1_combined)
            bn2 = BatchNormalization(momentum=0.9)(conv2, is_train)
            act2 = LeakyReLU()(bn2)

            # Flatten
            flatten = Flatten()(act2)

            ## 1 FC 
            fc1 = Dense(self.__fc_dim)(flatten)
            fc1 = LeakyReLU()(fc1)
            
            ## Dropout
            fc1_drop = Dropout(do_prob)(fc1)

            ## 2 FC
            logits = Dense(2)(fc1_drop)

            return logits
