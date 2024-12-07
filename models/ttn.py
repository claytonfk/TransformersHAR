# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:06:09 2023

@author: clayt
"""

import tensorflow as tf
from .transformer_layers import Encoder, AttentionWithContext, PositionalEncoding, SensorAttention
from tensorflow.keras.layers import  Dense, Conv1D, LSTM

class TTN(tf.keras.Model):
    
    # To implement deepnorm and subln
    def __init__(self, num_classes, num_channels, window_length, model_dim = 128, num_heads_spatial = 4, num_heads_temporal = 4, num_layers_spatial = 1, num_layers_temporal = 2, dropout_rate = 0.2):
        super(TTN, self).__init__()
        
        self.model_name = "ttn"

        
        self.model_dim = model_dim
        
        
        self.window_length = window_length
        self.num_channels = num_channels
        
        # Attention Block
        self.sensor_attention = SensorAttention(self.model_dim, (3, 3), (1, 1))
        self.conv_spatial = Conv1D(self.model_dim, 1, strides=1, padding = 'same')
        self.conv_temporal = Conv1D(self.model_dim, 1, strides=1, padding = 'same')
        self.layer_norm =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pos_encoding = PositionalEncoding(self.window_length, self.model_dim)
        
        self.awc_spatial = AttentionWithContext(self.model_dim)
        self.awc_temporal = AttentionWithContext(self.model_dim)
        
        
        
        # self.spatial_layers = [
        #                         Encoder(self.model_dim, self.model_dim, num_heads_spatial, dropout_rate, dropout_rate) for _ in range(num_layers_spatial)]
        
        # self.temporal_layers =  [
        #                         Encoder(self.model_dim, self.model_dim, num_heads_temporal, dropout_rate, dropout_rate) for _ in range(num_layers_temporal)]
        
        
        self.spatial_layers = [LSTM(self.model_dim, return_sequences=True) for _ in range(num_layers_spatial)]
        
        self.temporal_layers = [LSTM(self.model_dim, return_sequences=True) for _ in range(num_layers_temporal)]

        self.dense = Dense(num_classes)
        
        self.info = f"model_dim_{model_dim}_sh_{num_heads_spatial}_th_{num_heads_temporal}_ls_{num_layers_spatial}_lt_{num_layers_temporal}_do_{dropout_rate}"
        

    def call(self, inputs, training=False):
        # Forward pass
    
        # Attention Block
        x, _ = self.sensor_attention(inputs, training = training)
        # Spatial 
        spatial_x = self.conv_spatial(x)
        
        # Temporal
        temporal_x = self.conv_temporal(x) 
        temporal_x = self.pos_encoding(temporal_x)


        for layer in self.spatial_layers:
            spatial_x = layer(spatial_x, training=training)
        
        for layer in self.temporal_layers:
            temporal_x = layer(temporal_x, training=training)  
            

        #print(spatial_x.shape)
        #print(temporal_x.shape)
        spatial_x = self.awc_spatial(spatial_x)
        temporal_x = self.awc_temporal(temporal_x)

        
        x = tf.concat([spatial_x, temporal_x], axis = 1)
                
        x = self.dense(x)

        x = tf.nn.softmax(x)
        return x