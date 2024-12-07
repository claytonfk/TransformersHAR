import tensorflow as tf

class RBF(tf.keras.layers.Layer):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (tf.range(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return tf.reduce_sum(L2_distances) / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def call(self, X):
        L2_distances = tf.reduce_sum(tf.square(tf.linalg.pdist(X)), axis=-1)
        return tf.reduce_sum(tf.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]), axis=0)


class MMDLoss(tf.keras.losses.Loss):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def call(self, X, Y):
        K = self.kernel(tf.concat([X, Y], axis=0))

        X_size = X.shape[0]
        XX = tf.reduce_mean(K[:X_size, :X_size])
        XY = tf.reduce_mean(K[:X_size, X_size:])
        YY = tf.reduce_mean(K[X_size:, X_size:])
        return XX - 2 * XY + YY


class SubjectMMDLoss(tf.keras.losses.Loss):

    def __init__(self, N, kernel=RBF()):
        super().__init__()
        self.MMDLoss = MMDLoss(kernel)
        self.N = N

    def call(self, X, S):
        subjects = tf.unique(S)
        loss = tf.zeros(1)

        for i, s in enumerate(subjects):
            for t in subjects[(i+1):]:
                X_s = X[tf.where(S == s)[0], ...]
                X_t = X[tf.where(S == t)[0], ...]
                loss += self.MMDLoss(X_s, X_t)

        return loss / (self.N ** 2)
