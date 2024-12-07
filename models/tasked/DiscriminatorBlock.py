import tensorflow as tf
from tensorflow.keras import layers

class DiscriminatorBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, conv_kernel_size=5, conv_stride=2, fc_hidden_dim=128,
                 dropout_prob=0.5):
        super(DiscriminatorBlock, self).__init__()

        self.conv_layer = layers.Conv1D(out_channels,
                                        conv_kernel_size,
                                        strides=conv_stride,
                                        padding='same',
                                        data_format='channels_first')

        self.batch_norm = layers.BatchNormalization(axis=-1)

        self.dropout = layers.Dropout(dropout_prob)

    def call(self, x, training=False):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)

        return x
