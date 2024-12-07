# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 01:01:56 2023

@author: clayt
"""

import tensorflow as tf
from tensorflow.keras import layers

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = layers.Dense(inner_dim, use_bias=False)
        self.to_v = layers.Dense(inner_dim, use_bias=False)
        self.to_q = layers.Dense(inner_dim, use_bias=False)
        
        if self.project_out:
            self.to_out_dense = layers.Dense(dim)
            self.to_out_dropout = layers.Dropout(dropout)

    def call(self, x_qkv, training = False):
        b, n, _, h = x_qkv.shape[0], x_qkv.shape[1], x_qkv.shape[2], self.heads

        k = self.to_k(x_qkv)
        k = tf.reshape(k, (b, n, h, -1))
        k = tf.transpose(k, perm = [0, 2, 1, 3])

        v = self.to_v(x_qkv)
        v = tf.reshape(v, (b, n, h, -1))
        v = tf.transpose(v, perm = [0, 2, 1, 3])

        q = self.to_q(x_qkv) #tf.expand_dims(x_qkv[:, 0], axis = 1))
        #print(q.shape)
        q = tf.reshape(q, (b, n, h, -1))
        q = tf.transpose(q, perm = [0, 2, 1, 3])
        

        dots = tf.einsum('bhin,bhjd->bhij', q, k) * self.scale

        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = tf.reshape(out, (b, n, -1))
        if self.project_out:
            out = self.to_out_dense(out)
            out = self.to_out_dropout(out, training = training)
        return out



class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)

        if self.project_out:
            self.to_out_dense = layers.Dense(dim)
            self.to_out_dropout = layers.Dropout(dropout)

    def call(self, x, training = False):
        b, n, _, h = x.shape[0], x.shape[1], x.shape[2], self.heads

        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: tf.reshape(t, (b, n, h, -1)), qkv)
        
        k = tf.transpose(k, perm = [0, 2, 1, 3])
        v = tf.transpose(v, perm = [0, 2, 1, 3])
        q = tf.transpose(q, perm = [0, 2, 1, 3])
        
        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = tf.transpose(out, perm = [0, 2, 1, 3])
        out = tf.reshape(out, (b, n, -1))
        if self.project_out:
            out = self.to_out_dense(out)
            out = self.to_out_dropout(out, training = training)
        return out


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.Activation('gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, training = False):
        return self.net(x, training = training)


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training = False):
        x = self.attention(self.norm1(x), training = training) + x
        x = self.ffn(self.norm2(x), training = training) + x
        return x

class Transformer(tf.keras.layers.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = [TransformerLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]

    def call(self, x, training = False):
        for layer in self.layers:
            x = layer(x, training = training)
        return x
    

class MultiScaleTransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, small_dim=96, small_depth=4, small_heads=3, small_dim_head=32, small_mlp_dim=384,
                 large_dim=192, large_depth=1, large_heads=3, large_dim_head=64, large_mlp_dim=768,
                 cross_attn_depth=1, cross_attn_heads=3, dropout=0.):
        super(MultiScaleTransformerEncoder, self).__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)

        self.cross_attn_layers = []
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append([
                layers.Dense(large_dim),
                layers.Dense(small_dim),
                layers.LayerNormalization(epsilon=1e-6),
                CrossAttention(large_dim, heads=cross_attn_heads, dim_head=large_dim_head, dropout=dropout),
                layers.Dense(small_dim),
                layers.Dense(large_dim),
                layers.LayerNormalization(epsilon=1e-6),
                CrossAttention(small_dim, heads=cross_attn_heads, dim_head=small_dim_head, dropout=dropout),
            ])

    def call(self, xs, xl, training = False):

        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)

        for f_sl, g_ls, ln1, cross_attn_s, f_ls, g_sl, ln2, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class[:, tf.newaxis])
            cal_qkv = tf.concat([cal_q, x_small], axis=1)
            cal_out = cal_q + cross_attn_l(ln1(cal_qkv), training = training)
            cal_out = g_sl(cal_out)
            xl = tf.concat([cal_out, x_large], axis=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class[:, tf.newaxis])
            cal_qkv = tf.concat([cal_q, x_large], axis=1)
            cal_out = cal_q + cross_attn_s(ln2(cal_qkv), training = training)
            cal_out = g_ls(cal_out)
            xs = tf.concat([cal_out, x_small], axis=1)

        return xs, xl




class CrossViT(tf.keras.Model):
    def __init__(self, num_timesteps, num_channels, num_classes, patch_size_small, patch_size_large, small_dim=96,
                 large_dim=192, small_depth=1, large_depth=4, cross_attn_depth=1, multi_scale_enc_depth=3,
                 heads=3, pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super(CrossViT, self).__init__()

        # assert num_timesteps % patch_size_small[0] == 0
        # assert num_timesteps % patch_size_large[0] == 0
        
        # assert num_channels % patch_size_small[1] == 0
        # assert num_channels % patch_size_large[1] == 0
        
        self.patch_size_small = patch_size_small
        self.patch_size_large = patch_size_large
        
        num_patches_small = (num_timesteps // patch_size_small[0]) * (num_channels // patch_size_small[1])
        num_patches_large = (num_timesteps // patch_size_large[0]) * (num_channels // patch_size_large[1])

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.small_patch_dense = layers.Dense(small_dim)
        self.large_patch_dense = layers.Dense(large_dim)

        self.pos_embedding_small = self.add_weight("pos_embedding_small", shape=(1, num_patches_small + 1, small_dim))
        self.cls_token_small = self.add_weight("cls_token_small", shape=(1, 1, small_dim))
        self.dropout_small = layers.Dropout(emb_dropout)

        self.pos_embedding_large = self.add_weight("pos_embedding_large", shape=(1, num_patches_large + 1, large_dim))
        self.cls_token_large = self.add_weight("cls_token_large", shape=(1, 1, large_dim))
        self.dropout_large = layers.Dropout(emb_dropout)

        self.multi_scale_transformers = []
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim, small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=small_dim // heads,
                                                                              small_mlp_dim=small_dim * scale_dim,
                                                                              large_dim=large_dim, large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=large_dim // heads,
                                                                              large_mlp_dim=large_dim * scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.pool = pool

        self.mlp_head_small = tf.keras.Sequential([
            layers.LayerNormalization(axis=-1),
            layers.Dense(num_classes),
        ])

        self.mlp_head_large = tf.keras.Sequential([
            layers.LayerNormalization(axis=-1),
            layers.Dense(num_classes),
        ])
        
        self.model_name = 'cavit'
        
        patch_size_small_str = str(patch_size_small).replace(", ", "by").replace("[", "").replace("]", "")
        patch_size_large_str = str(patch_size_large).replace(", ", "by").replace("[", "").replace("]", "")

                
        self.info = f"pss_{patch_size_small_str}_psl_{patch_size_large_str}_sd_{small_dim}_ld_{large_dim}_nh_{heads}_msed_{multi_scale_enc_depth}_do_{dropout}"
        
    def get_patches(self, x, patch_size):
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size[0], patch_size[1], 1],
            strides=[1, patch_size[0], patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
        

    def call(self, x, training = False):
        x = tf.expand_dims(x, axis = 3)
        
        xs = self.get_patches(x, self.patch_size_small)
        xs = self.small_patch_dense(xs)
        b, n, _ = xs.shape

        cls_token_small = tf.tile(self.cls_token_small, [b, 1, 1])
        xs = tf.concat([cls_token_small, xs], axis=1)
        xs += self.pos_embedding_small[:, :(n + 1)]
        xs = self.dropout_small(xs, training = training)

        xl = self.get_patches(x, self.patch_size_large)
        xl = self.large_patch_dense(xl)
        b, n, _ = xl.shape

        cls_token_large = tf.tile(self.cls_token_large, [b, 1, 1])
        xl = tf.concat([cls_token_large, xl], axis=1)
        xl += self.pos_embedding_large[:, :(n + 1)]
        xl = self.dropout_large(xl, training = training)

        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xl = multi_scale_transformer(xs, xl, training = training)

        xs = tf.reduce_mean(xs, axis=1) if self.pool == 'mean' else xs[:, 0]
        xl = tf.reduce_mean(xl, axis=1) if self.pool == 'mean' else xl[:, 0]

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)
        x = xs + xl
        x = tf.nn.softmax(x)
        return x
    