# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 00:33:16 2023

@author: clayt
"""

import tensorflow as tf
import math
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


def rotate_every_two(x):
    # Split the input tensor into two parts along the last dimension
   
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]

    
    # Stack the two parts with a negative sign for the second part
    x = tf.stack((-x2, x1), axis=-1)
    
    # Flatten along the last two dimensions
    x = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2], x.shape[-2] * x.shape[-1]])
    
    return x

def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = tf.reshape(m, [-1, 1])  # Flatten the matrix
    m = tf.tile(m, [1, 2])  # Repeat all elements into the 2nd dimension
    m = tf.reshape(m, [dim0, -1])  # Reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    # print(x.shape)
    # print(sin.shape)
    # print(cos.shape)
    return (x * cos) + (rotate_every_two(x) * sin)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_timesteps, num_channels, num_retention_heads):
        super(PositionalEncoding, self).__init__()
        self.num_channels = num_channels
        self.num_retention_heads = num_retention_heads
        self.num_timesteps = num_timesteps
        self.build()
        

    def build(self):
        
        const = 10000
        angle = tf.linspace(0, 1, self.num_channels // self.num_retention_heads // 2)
        angle = const ** angle
        angle = 1.0 / angle
        
        angle = tf.expand_dims(angle, axis=-1)
        angle = tf.tile(angle, multiples=[1, 2])
        angle = tf.reshape(angle, shape=[-1])
        
        decay = tf.math.log(1 - 2 ** (-5 - tf.range(self.num_retention_heads, dtype=tf.float32)))

        self.angle = self.add_weight("angle", 
                                      shape=angle.shape,
                                      initializer=tf.constant_initializer(angle.numpy()),
                                      trainable=False)

        self.decay = self.add_weight("decay", 
                                      shape=decay.shape,
                                      initializer=tf.constant_initializer(decay.numpy()),
                                      trainable=False)
    def call(self, recurrent):
        if recurrent:
            sin = tf.sin(self.angle * (self.num_timesteps-1))
            cos = tf.cos(self.angle * (self.num_timesteps-1))
            return sin, cos, tf.exp(self.decay)
        else:
            index = tf.cast(tf.range(self.num_timesteps), dtype=self.decay.dtype)
            sin = tf.sin(tf.expand_dims(index, axis=1) * tf.expand_dims(self.angle, axis=0))
            cos = tf.cos(tf.expand_dims(index, axis=1) * tf.expand_dims(self.angle, axis=0))
            mask = tf.linalg.band_part(tf.ones((self.num_timesteps, self.num_timesteps), dtype=self.decay.dtype), -1, 0)
            bool_mask = tf.math.logical_not(tf.cast(mask, dtype=tf.bool))
            index_diff = tf.expand_dims(index, axis=1) - tf.expand_dims(index, axis=0)

            masked_index_diff = tf.where(bool_mask, tf.constant(float("inf"), dtype=self.decay.dtype), index_diff)
            masked_exp = tf.exp(masked_index_diff * self.decay[:, None, None])
            masked_exp = tf.where(tf.math.is_nan(masked_exp), tf.constant(0.0, dtype=self.decay.dtype), masked_exp)
            
            mask_sum = tf.reduce_sum(masked_exp, axis=-1, keepdims=True)
            sqrt_mask_sum = tf.sqrt(mask_sum)
            normalized_mask = masked_exp / sqrt_mask_sum
            return sin, cos, normalized_mask


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, ffn_dim, num_channels, activation_fn, final_dropout, activation_dropout, use_layernorm = False):
        super(FeedForwardNetwork, self).__init__()
        
        # Define the layers for the feedforward network
        self.dense1 = Dense(ffn_dim, activation = activation_fn)
        self.dense2 = Dense(num_channels, activation = activation_fn)
        
        
        self.act_dropout   = Dropout(activation_dropout)
        self.final_dropout = Dropout(final_dropout)
        
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.use_layernorm = use_layernorm


    def call(self, x, training = False):
        # Forward pass through the layers
        #x_shape = x.shape
        #print(x.shape)
        #x = tf.keras.layers.Flatten()(x)
        
        x = self.dense1(x)
        x = self.act_dropout(x, training = training)
        
        if self.use_layernorm:
            x = self.layernorm(x)
            
        x = self.dense2(x)
        #print(x.shape)
        #x = tf.reshape(x, x_shape)
        x = self.final_dropout(x, training = training)

        return x

   
class MultiScaleRetention(tf.keras.layers.Layer):
    # Possibly to do: weight initialization 
    def __init__(self, num_channels, activation_fn, value_factor, num_heads):
        super(MultiScaleRetention, self).__init__()
    
        
        self.factor = value_factor
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = self.num_channels * self.factor // num_heads
        self.key_dim = self.num_channels // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = Dense(num_channels, activation = activation_fn)
        self.k_proj = Dense(num_channels, activation = activation_fn)
        self.v_proj = Dense(num_channels*self.factor, activation = activation_fn)
        self.g_proj = Dense(num_channels*self.factor, activation = activation_fn)
        
        self.out_proj = Dense(num_channels, activation = activation_fn)

        self.group_norm = LayerNormalization(epsilon=1e-6)

    def parallel_forward(self, qr, kr, v, mask, get_attention_scores = False):
        bsz = tf.shape(v)[0]
        tgt_len = tf.shape(v)[1]
        
        vr = tf.reshape(v, (bsz, tgt_len, self.num_heads, self.head_dim))
        vr = tf.transpose(vr, perm=[0, 2, 1, 3])
        #print("vr shape", vr.shape)
        
        qk_mat = tf.matmul(qr, tf.transpose(kr, perm=[0, 1, 3, 2]))  # bsz * m * tgt_len * tgt_len
        #print("qk_mat shape", qk_mat.shape)
        
        qk_mat = qk_mat * mask
        #print("qk_mat shape", qk_mat.shape)
        
        qk_mat_sum = tf.math.reduce_sum(qk_mat, axis=-1, keepdims=True)
        qk_mat_sum = tf.math.abs(qk_mat_sum)
        qk_mat_sum = tf.clip_by_value(qk_mat_sum, clip_value_min = 1, clip_value_max = tf.float32.max)
        qk_mat = qk_mat / (qk_mat_sum + 1e-6)
        

        #print("qk_mat shape", qk_mat.shape)
        
        output = tf.matmul(qk_mat, vr)
        #print("output shape", output.shape)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        

        #print("output shape", output.shape)
        
        if get_attention_scores:
            return output, output
        
        return output

    
    def recurrent_forward(self, qr, kr, v, decay, prev_kv, prev_scale):
        bsz = tf.shape(v)[0]

        v = tf.reshape(v, (bsz, self.num_heads, self.head_dim, 1))

        kv = kr * v

        if not isinstance(prev_scale, int):
            prev_scale_sqrt = tf.math.sqrt(prev_scale)
            scale = prev_scale * decay + 1
            scale_sqrt = tf.math.sqrt(scale)
            
            
            prev_scale_sqrt = tf.reshape(prev_scale_sqrt, ( -1, 1, 1))
            decay = tf.reshape(decay, (-1, 1, 1))
            scale_sqrt = tf.reshape(scale_sqrt, (-1, 1, 1))
            


            p1 = prev_kv * (prev_scale_sqrt * decay / scale_sqrt)

            p2 = kv / scale_sqrt
            kv = p1 + p2

        else:
            scale = tf.ones_like(decay)

        # Sum along the last dimension (dimension 3)
        output = tf.reduce_sum(qr * kv, axis=3)

        return output, kv, scale

    def call(self, x, rel_pos, prev_kv = -1, prev_scale = 1, force_recurrent = False, get_attention_scores = False):

            
        bsz = tf.shape(x)[0]
        tgt_len = tf.shape(x)[1]
        sin, cos, inner_mask = rel_pos
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)
        
        k *= self.scaling
        q = tf.reshape(q, (bsz, tgt_len, self.num_heads, self.key_dim))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (bsz, tgt_len, self.num_heads, self.key_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        
        if force_recurrent:
            output, kv, scale = self.recurrent_forward(qr, kr, v, inner_mask, prev_kv, prev_scale)
        else:  
            if get_attention_scores:
                output, att = self.parallel_forward(qr, kr, v, inner_mask, get_attention_scores = True)
            else:
                output = self.parallel_forward(qr, kr, v, inner_mask)
        
        #output = self.group_norm(output)
        output = tf.reshape(output, (bsz, tgt_len, self.head_dim * self.num_heads))
        output = self.group_norm(output)

        g = g * tf.sigmoid(g) # Swish
        output = g * output
        
        output = self.out_proj(output)
        
        if force_recurrent:
            return output, kv, scale
        
        # if get_attention_scores:
        #     return output, att
        
        
        return output

    
    
class DecoderLayer(tf.keras.layers.Layer):
    # To implement: Drop Path
    def __init__(self, dropout, num_channels, value_factor, num_heads, normalize_before, num_decoder_layers, ffn_dim, activation_fn, 
                 final_dropout, activation_dropout, deepnorm = False, use_ffn_layernorm = False):
        super(DecoderLayer, self).__init__()
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.normalize_before = normalize_before
        self.deepnorm = deepnorm
        self.num_decoder_layers = num_decoder_layers

        # Build the FFN, Retention
        
        self.retention_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        if deepnorm:
            self.alpha = math.pow(2.0 * num_decoder_layers, 0.25)
        else:
            self.alpha = 1.0
            
        self.ffn = FeedForwardNetwork(ffn_dim, num_channels, activation_fn, final_dropout, activation_dropout, use_layernorm = use_ffn_layernorm)
        self.retention = MultiScaleRetention(num_channels, activation_fn, value_factor, num_heads)
        
    def residual_connection(self, x, residual):
        return residual * self.alpha + x
    
    def call(self, x, retention_rel_pos, prev_kv = -1, prev_scale = 1,  training = False, force_recurrent = False, get_attention_scores = False):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)
            
        x = self.retention(
            x,
            rel_pos=retention_rel_pos,
            prev_kv=prev_kv, 
            prev_scale=prev_scale,
            force_recurrent=force_recurrent,
            get_attention_scores = get_attention_scores
        )
        
        if force_recurrent:
            x, kv, scale = x
        
        
        x = self.dropout_layer(x, training = training)
        
        # print(x.shape)
        # print(residual.shape)
        x = self.residual_connection(x, residual)
        
        if not self.normalize_before:
            x = self.retention_layer_norm(x)
            
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
            
        x = self.ffn(x, training = training)
        x = self.residual_connection(x, residual)
        
        if not self.normalize_before:
            x = self.final_layer_norm(x)
            
        if force_recurrent:
            return x, kv, scale

        return x
        

class RetNetDecoder(tf.keras.Model):
    
    # To implement deepnorm and subln
    def __init__(self, num_classes, num_channels, num_timesteps, dropout, ffn_dim, num_decoder_layers, num_retention_heads, 
                 normalize_before = True, decoder_normalize_before  = True, deepnorm = False, use_ffn_layernorm = True, 
                 activation_fn = 'gelu', value_factor = 1, conv_filters = 128, conv_kernel = 2):
        super(RetNetDecoder, self).__init__()
        self.dropout_module = tf.keras.layers.Dropout(dropout)
        self.model_name = "retnet"

        self.decoder_normalize_before = decoder_normalize_before
        self.num_channels = conv_filters
        
        
        self.num_timesteps = num_timesteps
        self.num_retention_heads = num_retention_heads
        self.conv      = tf.keras.layers.Conv1D(conv_filters, conv_kernel, strides=1, padding = 'same')
        self.retlayers =  []
        
        for i in range(num_decoder_layers):
            layer = DecoderLayer(dropout, self.num_channels, value_factor, num_retention_heads, normalize_before, num_decoder_layers,
                                 ffn_dim, activation_fn, dropout, dropout, deepnorm = deepnorm, 
                                 use_ffn_layernorm = use_ffn_layernorm)
            self.retlayers.append(layer)

        self.num_layers = len(self.retlayers)

        if self.decoder_normalize_before:
            self.layer_norm =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.layer_norm = None
            

        self.retnet_rel_pos = PositionalEncoding(self.num_timesteps, self.num_channels, num_retention_heads)
        
        self.classifier = Dense(num_classes, activation = activation_fn)
        
        self.info = f"ffndim_{ffn_dim}_dl_{num_decoder_layers}_rh_{num_retention_heads}_cf_{conv_filters}_ck_{conv_kernel}_do_{dropout}"
                

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)
    
    def call_parallel(self, x, get_attention_scores = False):

        # relative position
        
        retention_rel_pos = self.retnet_rel_pos.call(False)
        
        # decoder layers
        inner_states = [x]
        att_scores = []
        
        for idx, layer in enumerate(self.retlayers):

            x = layer(x, retention_rel_pos, None, training = True, force_recurrent = False, get_attention_scores = get_attention_scores)
            
            if get_attention_scores:
                x, att = x
            
            inner_states.append(x)
            att_scores.append(x)
            
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        
        x = tf.reshape(x, (-1, x.shape[1]*x.shape[2]))
        x = self.classifier(x)
        x = tf.nn.softmax(x)
        
        if get_attention_scores:
            return x, att_scores

        return x

    # @tf.function
    # def call_recurrent(self, inputs, get_attention_scores = False):
    #     seq_len = inputs.shape[1]
    #     incremental_state = [{}]*len(self.retlayers)
    #     retention_rel_pos = self.retnet_rel_pos.call(True)
    #     combined_x = []
        
    #     for i in range(0, seq_len):
    #         sub_x = inputs[:, i, :]
    #         sub_x = sub_x[:, None, :]
    #         x = sub_x
            

    #         att_scores = []

    #         for idx, layer in enumerate(self.retlayers):
    #             x = layer(x, retention_rel_pos, incremental_state[idx], training = False, force_recurrent = True, 
    #                       get_attention_scores= get_attention_scores)
                
    #             if get_attention_scores:
    #                 x, att = x
                
    #             att_scores.append(x)
                

                
    #         combined_x.append(x)

    #     x = tf.concat(combined_x, axis = 1)
        
    #     if self.layer_norm is not None:
    #         x = self.layer_norm(x)
    #     x = tf.reshape(x, (-1, x.shape[1]*x.shape[2]))
    #     x = self.classifier(x)
    #     x = tf.nn.softmax(x)
        
    #     return x

    def recurrent_loop(self, inputs, i, retention_rel_pos, cx, last_kvs, last_scales):
        x = inputs[:, i, :]
        x = x[:, None, :]
        
        kvs    = [-1]*len(self.retlayers)
        scales = [1]*len(self.retlayers)
        for idx, layer in enumerate(self.retlayers):
            x, kv, scale = layer(x, retention_rel_pos, last_kvs[idx], last_scales[idx], training = False, 
                                 force_recurrent = True, get_attention_scores = False)
            
            kvs[idx] = kv
            scales[idx] = scale
            

        return [inputs, i + 1, retention_rel_pos, tf.concat([cx, x], axis=1), kvs, scales]
        
    

    def call_recurrent(self, inputs, get_attention_scores = False):
        seq_len           = inputs.shape[1]
        last_kvs          = [-1]*len(self.retlayers)
        last_scales       = [1]*len(self.retlayers)
        retention_rel_pos = self.retnet_rel_pos.call(True)
        combined_x        = tf.zeros([inputs.shape[0], 1, self.num_channels])

        loop_index        = tf.constant(0)
        loop_condition    = lambda a, i, b, c, d, e: tf.less(i, seq_len)
        loop_body         = self.recurrent_loop
        loop_args         = [inputs, loop_index, retention_rel_pos, combined_x, last_kvs, last_scales]
        
        _, _, _, x, _, _ = tf.while_loop(loop_condition, loop_body, loop_args)
        x = x[:, 1:, :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
            
        x = tf.reshape(x, (-1, x.shape[1]*x.shape[2]))
        x = self.classifier(x)
        x = tf.nn.softmax(x)
        
        return x
    
            
    def call(self, x, incremental_state=None, training = False, force_recurrent = False, get_attention_scores = False):
        x = self.conv(x)
        if force_recurrent:
            x = self.call_recurrent(x, get_attention_scores = get_attention_scores)
        else:
            x = self.call_parallel(x, get_attention_scores = get_attention_scores)
            
        return x