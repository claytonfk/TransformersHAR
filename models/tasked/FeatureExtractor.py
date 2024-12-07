import tensorflow as tf
from tensorflow.keras import layers, models

from .spatial_attention_block import SpatialAttentionBlock
from .positionalembedding2d_t import positionalencoding2d


class ConvBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv1D(out_channels, kernel_size, strides=stride, padding='same', data_format='channels_first')
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.relu(x)
        return x

class FeatureExtractor(models.Model):
    def __init__(self, Cs, W, S):
        super(FeatureExtractor, self).__init__()
        self.Cs = Cs
        self.W = W
        self.S = S

        self.conv_blocks = [ConvBlock(Cs, 32, kernel_size=3, stride=1) for _ in range(S)]

        self.positional_encoding = positionalencoding2d(32, W, S)

        self.sa_block1 = SpatialAttentionBlock(32, W, S, 8)
        self.sa_block2 = SpatialAttentionBlock(64, W//2, S, 8)
        self.sa_block3 = SpatialAttentionBlock(128, W//4, S, 8)

        self.avg_pool = layers.AveragePooling2D((1, S))

    def call(self, x, training=False):
        conv_outputs = [conv_block(x[:, self.Cs*i:self.Cs*(i+1), :], training=training)[:, :, tf.newaxis] for i, conv_block in enumerate(self.conv_blocks)]
        conv_outputs = tf.concat(conv_outputs, axis=2)

        positional_encoded = conv_outputs + tf.expand_dims(self.positional_encoding, axis=0) 

        sa_block1_out = self.sa_block1(tf.transpose(positional_encoded, (0,1,3,2)), training=training)
        sa_block2_out = self.sa_block2(sa_block1_out, training=training)
        
        sa_block3_out = self.sa_block3(sa_block2_out, training=training)
        


        sa_block3_out = tf.transpose(sa_block3_out, [0, 2, 3, 1])

        avg_pooled = self.avg_pool(sa_block3_out, training=training)


        avg_pooled = tf.transpose(avg_pooled, [0, 3, 1, 2])

        return tf.squeeze(avg_pooled, axis=-1)
