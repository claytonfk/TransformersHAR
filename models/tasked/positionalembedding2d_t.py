import math
import numpy as np
import tensorflow as tf

def positionalencoding1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    position = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0., d_model, 2.) * -(math.log(10000.0) / d_model))
    pos_enc = tf.zeros((length, d_model))
    pos_enc[:, 0::2] = tf.sin(position * div_term)
    pos_enc[:, 1::2] = tf.cos(position * div_term)

    return pos_enc

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = tf.zeros([d_model, height, width])
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = tf.exp(tf.range(0., d_model, 2) *
                      -(np.log(10000.0) / d_model))
    pos_w = tf.range(0., width)[:, tf.newaxis]
    pos_h = tf.range(0., height)[:, tf.newaxis]
    pe = tf.Variable(pe)
    pe[0:d_model:2, :, :].assign(tf.sin(pos_w * div_term).numpy().T[:, tf.newaxis, :].repeat(height, axis=1))
    pe[1:d_model:2, :, :].assign(tf.cos(pos_w * div_term).numpy().T[:, tf.newaxis, :].repeat(height, axis=1))
    pe[d_model::2, :, :].assign(tf.sin(pos_h * div_term).numpy().T[:, :, tf.newaxis].repeat(width, axis=2))
    pe[d_model + 1::2, :, :].assign(tf.cos(pos_h * div_term).numpy().T[:, :, tf.newaxis].repeat(width, axis=2))
    return tf.transpose(pe, (0,2,1))

