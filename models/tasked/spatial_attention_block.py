import tensorflow as tf
from tensorflow.keras import layers

from .head import SpatialAttentionHead

class SpatialAttentionBlock(layers.Layer):
    def __init__(self, C, W, Sources, num_heads=8):
        super(SpatialAttentionBlock, self).__init__()
        self.C = C
        self.W = W
        self.Sources = Sources
        self.num_heads = num_heads

        self.conv1 = layers.Conv2D(2*C, (1, 1), strides=(2, 1), data_format='channels_first')
        #self.bn1 = layers.BatchNormalization(axis=-1)

        self.bn2 = layers.BatchNormalization(axis=-1)
        self.spatial_attention_heads = [SpatialAttentionHead(C // num_heads, 2 * C // num_heads) for _ in range(num_heads)]
        #self.bn3 = layers.BatchNormalization(axis=-1)
        self.dropout = layers.Dropout(0.5)
        self.conv2 = layers.Conv2D(2*C, (9, 1), strides=(2, 1), padding='same', data_format='channels_first')
        #self.bn4 = layers.BatchNormalization(axis=-1)
        self.relu = layers.ReLU()

    def call(self, inputs, training=None):
        #batch_size = tf.shape(inputs)[0]
        
        
        x = tf.reshape(tf.transpose(inputs, perm=[0, 3, 2, 1]), [-1, self.C * self.Sources, self.W])
        x = self.bn2(x, training=training)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1]), [-1, self.W, self.C, self.Sources])
        x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2]), [-1, self.W, self.Sources, self.num_heads, self.C // self.num_heads])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        # #print(x.shape)
        x = [self.spatial_attention_heads[i](x[:, :, i, :, :], training=training) for i in range(self.num_heads)]
        x = tf.concat(x, axis=2)
        # #print(x.shape)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        #x = self.bn3(x, training=training)
        x = self.dropout(self.relu(x), training=training)
        x = self.conv2(x, training=training)
        #x = self.bn4(x, training=training)
        x = self.relu(x)

        y = self.conv1(inputs, training=training)
        #y = self.bn1(y, training=training)
        y = self.relu(y)

        return y + x
