# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:11:38 2023

@author: clayt
"""

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout, Dense

    
class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(InceptionModule, self).__init__()
        
        # Path 1
        self.conv1_1 = Conv2D(num_filters, (1, 1), strides=(1, 1), padding = 'same', activation = 'elu')

        # Path 2
        self.conv1_2 = Conv2D(num_filters, (1, 1), strides=(1, 1), padding = 'same', activation = 'elu')
        self.conv2_2 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding = 'same', activation = 'elu')
        
        # Path 3
        self.conv1_3 = Conv2D(num_filters, (1, 1), strides=(1, 1), padding = 'same', activation = 'elu')
        self.conv2_3 = Conv2D(num_filters, (1, 5), strides=(1, 1), padding = 'same', activation = 'elu')

        # Path 4
        self.max_pool = MaxPooling2D(pool_size = (1, 2), strides=(1, 1), padding='same')
        self.conv1_4 = Conv2D(num_filters, (1, 1), strides=(1, 1), padding = 'same', activation = 'elu')
        
        self.max_pool_final = MaxPooling2D(pool_size = (1, 2), strides=(1, 2), padding='same')

        
    def call(self, x, training = False):
        # Path 1
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis = 3)
        
        x1 = self.conv1_1(x)
        # Path 2
        x2 = self.conv1_2(x)
        x2 = self.conv2_2(x2)
        
        # Path 3
        x3 = self.conv1_3(x)
        x3 = self.conv2_3(x3)
        
        # Path 4
        x4 = self.max_pool(x)
        x4 = self.conv1_4(x4)
        
        x = tf.concat([x1, x2, x3, x4], axis = -1)
        
        x = self.max_pool_final(x)
        
        return x
    

    
class InnoHAR(tf.keras.Model):
    def __init__(self, num_classes, conv_num_filters = 128, gru_units = 64, dropout_rate = 0.2, unroll = False):
        super(InnoHAR, self).__init__()
        self.model_name = "InnoHAR"
        self.num_classes = num_classes
        self.filters = conv_num_filters
        
        self.gru1 = tf.keras.layers.GRU(gru_units, unroll = unroll, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(gru_units, unroll = unroll, return_sequences=False)
        self.i1 = InceptionModule(conv_num_filters)
        self.i2 = InceptionModule(conv_num_filters)
        self.i3 = InceptionModule(conv_num_filters)
        self.i4 = InceptionModule(conv_num_filters)
        self.dropout = Dropout(dropout_rate)

        self.out = Dense(num_classes) 
        self.max_pool = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding='same')
        
        self.info =  self.info = f"convfilters_{conv_num_filters}_gruunits_{gru_units}_do_{dropout_rate}"
        
        
    def call(self, x, training = False):
        x = tf.expand_dims(x, axis = 1)
        x = self.i1(x)
        x = self.i2(x)
        x = self.i3(x)
        x =  tf.nn.max_pool(x, ksize=[1, 1, int(x.shape[2]/2), 1], strides =[1, 1, 1, 1], padding='SAME')
        x = self.i4(x)
        x =  tf.nn.max_pool(x, ksize=[1, 1, int(x.shape[2]/2), 1], strides =[1, 1, 1, 1], padding='SAME')
        x = tf.reshape(x, [-1, 1, 4*self.filters*x.shape[2]])
        

        x = self.dropout(x, training = training)
        x = self.gru1(x)
        x = self.gru2(x)
        x = self.out(x)
        x = tf.nn.softmax(x)
        return x

