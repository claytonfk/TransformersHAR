# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:01:01 2023

@author: clayt
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Bidirectional, LSTM, Dropout, Dense

class ResidualBiLSTM(tf.keras.Model):
    # To implement deepnorm and subln
    def __init__(self, num_classes, dropout_rate = 0.5, conv_filters = 32,  lstm_units = 64, unroll = False):
        super(ResidualBiLSTM, self).__init__()
        self.model_name = "resbilstm"
        self.conv1 = Conv2D(conv_filters, (2, 2), strides=(2, 2), padding = 'same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(conv_filters, (2, 2), strides=(1, 1), padding = 'same')
        self.bn2 = BatchNormalization()
        
        self.conv3 = Conv2D(conv_filters, (1, 1), strides=(2, 2), padding = 'same')
        self.bn3 = BatchNormalization()
        
        self.bilstm = Bidirectional(LSTM(lstm_units, unroll = unroll))
        self.dense = Dense(num_classes)

        self.dropout = Dropout(dropout_rate)
        
        self.info = f"convfilters_{conv_filters}_lstmunits_{lstm_units}_do_{dropout_rate}"
        
    def call(self, inputs, training = False):
        # Forward pass
    
        # Path 1
        inputs = tf.expand_dims(inputs, axis = 3)
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1, training = training)
        x1 = self.relu(x1)
        
        x1 = self.conv2(x1)
        x1 = self.bn2(x1, training = training)
        
        
        # Path 2
        x2 = self.conv3(inputs)
        x2 = self.bn3(x2, training = training)
    
        x = x1 + x2

        x = self.relu(x)        
        x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

        
        x = self.bilstm(x)
        
        x = self.dropout(x, training = training)
        # Pass through the Dense layer
        x = self.dense(x)
        x = tf.nn.softmax(x)
        
        return x
