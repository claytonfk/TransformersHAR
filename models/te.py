# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:07:44 2023

@author: clayt
"""

import tensorflow as tf
from .transformer_layers import Encoder, AttentionWithContext, PositionalEncoding, SensorAttention, Dropout
from tensorflow.keras.layers import  Dense, Conv1D, LSTM


class TE(tf.keras.Model):
    
    # To implement deepnorm and subln
    def __init__(self, num_classes, num_channels, window_length, num_heads_spatial = 4, num_layers_spatial = 2, model_dim = 128, num_filters = 128, dropout_rate = 0.1):
        super(TE, self).__init__()
        
        self.model_name = "te"
        num_heads_spatial = num_heads_spatial
        num_layers_spatial = num_layers_spatial
        
        self.model_dim = model_dim
        self.num_filters = num_filters
        
        
        self.window_length = window_length
        self.num_channels = num_channels
        
        # Attention Block
        self.sensor_attention = SensorAttention(self.num_filters, (3, 3), (2, 2))
        self.conv_spatial = Conv1D(self.model_dim, 1, strides=1, padding = 'same', activation = 'relu')
        self.pos_encoding = PositionalEncoding(self.window_length, self.model_dim)
        
        self.awc_spatial = AttentionWithContext(self.model_dim)

        self.spatial_layers = [Encoder(self.model_dim, self.model_dim, num_heads_spatial, dropout_rate, dropout_rate) for _ in range(num_layers_spatial)]
        
        
        #self.spatial_layers = [LSTM(self.model_dim, return_sequences=True) for _ in range(num_layers_spatial)]


        self.dropout_0 = Dropout(rate=dropout_rate)
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.pre_dense = Dense(num_classes*4,  activation="relu")
        self.dense = Dense(num_classes)
        
        
        self.info = f"hs_{num_heads_spatial}_ls_{num_layers_spatial}_dim_{model_dim}_nf_{num_filters}_do_{dropout_rate}"
        

    def call(self, inputs, training=False):
        # Forward pass
    
        # Attention Block
        x, _ = self.sensor_attention(inputs)
        # Spatial 
        x = self.conv_spatial(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout_0(x, training = training)
        
        #print(x.shape)


        for layer in self.spatial_layers:
            x = layer(x, training=training)
            
        #print(x.shape)
        


        x = self.awc_spatial(x)

        
        x = self.pre_dense(x)
        x = self.dropout_1(x, training = training)
        x = self.dense(x)
        x = tf.nn.softmax(x)
        
        return x