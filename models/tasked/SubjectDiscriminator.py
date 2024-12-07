import tensorflow as tf
from tensorflow.keras import layers, models

from .DiscriminatorBlock import DiscriminatorBlock

class SubjectDiscriminator(models.Model):
    def __init__(self, N=10, W=64):
        super(SubjectDiscriminator, self).__init__()

        self.discriminator_block1 = DiscriminatorBlock(256, 32, conv_kernel_size=5, conv_stride=2)

        self.discriminator_block2 = DiscriminatorBlock(32, 64, conv_kernel_size=5, conv_stride=2)

        self.discriminator_block3 = DiscriminatorBlock(64, 128, conv_kernel_size=5, conv_stride=2)

        self.fc1 = layers.Dense(10)
        self.fc2 = layers.Dense(N + 1, activation='softmax')

    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.discriminator_block1(x, training=training))

        x = tf.nn.leaky_relu(self.discriminator_block2(x, training=training))

        x = tf.nn.leaky_relu(self.discriminator_block3(x, training=training))

        x = tf.reshape(x, (tf.shape(x)[0], -1))

        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)

        return x
