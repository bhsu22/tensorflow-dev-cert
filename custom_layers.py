"""
Exercise 12.12
"""
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow import keras


class CustomLayerNormalization(keras.layers.Layer):
    def __init__(self, epsilon=0.001, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 0.001

    def build(self, batch_input_shape):
        shape = batch_input_shape[-1:]
        self.alpha = self.add_weight(
            name="alpha", initializer="ones", shape=batch_input_shape[-1:]
        )
        self.beta = self.add_weight(
            name="beta", initializer="zeros", shape=batch_input_shape[-1:]
        )
        super().build(batch_input_shape)

    def call(self, inputs):
        mu, sigma2 = tf.nn.moments(inputs, axes=-1, keepdims=True)

        normed_inputs = (
            self.alpha * (inputs - mu) / tf.sqrt(sigma2 + self.epsilon) + self.beta
        )
        return normed_inputs


if __name__ == "__main__":
    test1 = tf.random.normal((10, 10))

    CLN = CustomLayerNormalization()
    LN = LayerNormalization()

    cln_test1 = CLN(test1)
    ln_test1 = LN(test1)

    error = tf.reduce_mean(tf.losses.mean_absolute_error(cln_test1, ln_test1))
    print(error.numpy() < 1e-7)
