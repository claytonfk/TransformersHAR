import tensorflow as tf
from tensorflow.keras import layers, models

class ActivityClassifier(models.Model):
    def __init__(self, W, na):
        super(ActivityClassifier, self).__init__()

        self.avg_pool = layers.AveragePooling1D(pool_size=W//8)

        self.fc = layers.Dense(na)

    def call(self, x, training=False):
        x = self.avg_pool(x)

        x = tf.reshape(x, (tf.shape(x)[0], -1))

        x = self.fc(x)
        x = tf.nn.softmax(x)

        return x
