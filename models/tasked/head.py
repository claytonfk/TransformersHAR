import tensorflow as tf
from tensorflow.keras import layers
#import tensorflow_probability as tfp

from .DropConnect import DropConnect

import warnings


class WeightNorm(tf.keras.layers.Wrapper):
  """Layer wrapper to decouple magnitude and direction of the layer's weights.

  This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem. It has an optional data-dependent
  initialization scheme, in which initial values of weights are set as functions
  of the first minibatch of data. Both the weight normalization and data-
  dependent initialization are described in [Salimans and Kingma (2016)][1].

  #### Example

  ```python
    net = WeightNorm(tf_keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf_keras.layers.Conv2DTranspose(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf_keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf_keras.layers.Dense(num_classes),
                     data_init=True)(net)
  ```

  #### References

  [1]: Tim Salimans and Diederik P. Kingma. Weight Normalization: A Simple
       Reparameterization to Accelerate Training of Deep Neural Networks. In
       _30th Conference on Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1602.07868
  """

  def __init__(self, layer, data_init=True, **kwargs):
    """Initialize WeightNorm wrapper.

    Args:
      layer: A `tf_keras.layers.Layer` instance. Supported layer types are
        `Dense`, `Conv2D`, and `Conv2DTranspose`. Layers with multiple inputs
        are not supported.
      data_init: `bool`, if `True` use data dependent variable initialization.
      **kwargs: Additional keyword args passed to `tf_keras.layers.Wrapper`.

    Raises:
      ValueError: If `layer` is not a `tf_keras.layers.Layer` instance.

    """
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          'Please initialize `WeightNorm` layer with a `tf_keras.layers.Layer` '
          'instance. You passed: {input}'.format(input=layer))

    layer_type = type(layer).__name__
    if layer_type not in ['Dense', 'Conv2D', 'Conv2DTranspose']:
      warnings.warn('`WeightNorm` is tested only for `Dense`, `Conv2D`, and '
                    '`Conv2DTranspose` layers. You passed a layer of type `{}`'
                    .format(layer_type))

    super(WeightNorm, self).__init__(layer, **kwargs)

    self.data_init = data_init
    self._track_trackable(layer, name='layer')
    self.filter_axis = -2 if layer_type == 'Conv2DTranspose' else -1

  def _compute_weights(self):
    """Generate weights with normalization."""
    # Determine the axis along which to expand `g` so that `g` broadcasts to
    # the shape of `v`.
    new_axis = -self.filter_axis - 3

    # `self.kernel_norm_axes` is determined by `self.filter_axis` and the rank
    # of the layer kernel, and is thus statically known.
    self.layer.kernel = tf.nn.l2_normalize(
        self.v, axis=self.kernel_norm_axes) * tf.expand_dims(self.g, new_axis)

  def _init_norm(self):
    """Set the norm of the weight vector."""
    kernel_norm = tf.sqrt(
        tf.reduce_sum(tf.square(self.v), axis=self.kernel_norm_axes))
    self.g.assign(kernel_norm)

  def _data_dep_init(self, inputs):
    """Data dependent initialization."""
    # Normalize kernel first so that calling the layer calculates
    # `tf.dot(v, x)/tf.norm(v)` as in (5) in ([Salimans and Kingma, 2016][1]).
    self._compute_weights()

    activation = self.layer.activation
    self.layer.activation = None

    use_bias = self.layer.bias is not None
    if use_bias:
      bias = self.layer.bias
      self.layer.bias = tf.zeros_like(bias)

    # Since the bias is initialized as zero, setting the activation to zero and
    # calling the initialized layer (with normalized kernel) yields the correct
    # computation ((5) in Salimans and Kingma (2016))
    x_init = self.layer(inputs)
    norm_axes_out = list(range(x_init.shape.rank - 1))
    m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
    scale_init = 1. / tf.sqrt(v_init + 1e-10)

    self.g.assign(self.g * scale_init)
    if use_bias:
      self.layer.bias = bias
      self.layer.bias.assign(-m_init * scale_init)
    self.layer.activation = activation

  def build(self, input_shape=None):
    """Build `Layer`.

    Args:
      input_shape: The shape of the input to `self.layer`.

    Raises:
      ValueError: If `Layer` does not contain a `kernel` of weights
    """

    input_shape = tf.TensorShape(input_shape).as_list()
    input_shape[0] = None
    self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

    if not self.layer.built:
      self.layer.build(input_shape)

      if not hasattr(self.layer, 'kernel'):
        raise ValueError('`WeightNorm` must wrap a layer that'
                         ' contains a `kernel` for weights')

      kernel_norm_axes = list(range(self.layer.kernel.shape.rank))
      kernel_norm_axes.pop(self.filter_axis)
      # Convert `kernel_norm_axes` from a list to a constant Tensor to allow
      # TF checkpoint saving.
      self.kernel_norm_axes = tf.constant(kernel_norm_axes)

      self.v = self.layer.kernel

      # to avoid a duplicate `kernel` variable after `build` is called
      self.layer.kernel = None
      self.g = self.add_weight(
          name='g',
          shape=(int(self.v.shape[self.filter_axis]),),
          initializer='ones',
          dtype=self.v.dtype,
          trainable=True)
      self.initialized = self.add_weight(
          name='initialized',
          dtype=tf.bool,
          trainable=False)
      self.initialized.assign(False)

    super(WeightNorm, self).build()

  def call(self, inputs):
    """Call `Layer`."""
    if not self.initialized:
      if self.data_init:
        self._data_dep_init(inputs)
      else:
        # initialize `g` as the norm of the initialized kernel
        self._init_norm()

      self.initialized.assign(True)

    self._compute_weights()
    output = self.layer(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())


class SpatialAttentionHead(layers.Layer):
    def __init__(self, C_in, C_out):
        super(SpatialAttentionHead, self).__init__()
        self.C_dim = C_in

        self.Q = layers.Conv2D(C_in, (1, 1), strides=(1, 1), data_format='channels_first')
        self.K = layers.Conv2D(C_in, (1, 1), strides=(1, 1), data_format='channels_first')
        self.V = layers.Conv2D(C_in, (1, 1), strides=(1, 1), data_format='channels_first')

        self.dc = DropConnect(0.5)
        self.weight_norm = layers.Conv2D(C_in, (1,1), (1,1))
        self.conv1 = layers.Conv2D(C_out, (1, 1), strides=(1, 1), data_format='channels_first')

    def key_query_function(self, q, k):
        QK_T = tf.matmul(tf.transpose(q, perm=[0, 3, 1, 2]), tf.transpose(k, perm=[0, 3, 2, 1]))
        QK_T = tf.transpose(QK_T, perm=[0, 2, 3, 1])
        sqrt_d_k = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        return QK_T / sqrt_d_k

    def call(self, inputs, training=None):
        inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        qk = tf.nn.softmax(self.key_query_function(q, k), axis=-1)

        dc = self.dc(qk, training=training)
        wn = tf.transpose(self.weight_norm(tf.transpose(dc, (0,3,1,2))), (0,2,3,1))

        matmul = tf.matmul(tf.transpose(wn, perm=[0, 3, 1, 2]), tf.transpose(v, perm=[0, 3, 1, 2]))
        output = self.conv1(tf.transpose(matmul,(0,2,3,1)))

        output = tf.transpose(output, perm=[0, 2, 1, 3])

        return output


