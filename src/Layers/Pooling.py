import tensorflow as tf


class ChannelWiseGlobalAveragePooling(tf.keras.layers.Layer):

    def call(self, events):
        """Returns the feature-wise avarage over all the particles in the event"""
        return tf.reduce_mean(events, axis=1)

    def compute_output_shape(self, input_shape):
        batch_size, particles_features = input_shape[0], input_shape[-1]
        return batch_size, particles_features
