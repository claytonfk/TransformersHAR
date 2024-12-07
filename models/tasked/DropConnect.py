import tensorflow as tf

class DropConnect(tf.keras.layers.Layer):
    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.drop_rate = drop_rate

    def call(self, inputs, training=None):
        if training:
            mask = tf.keras.backend.random_bernoulli(shape=inputs.shape, p=1.0 - self.drop_rate)
            return inputs * tf.cast(mask, dtype=inputs.dtype)
        return inputs