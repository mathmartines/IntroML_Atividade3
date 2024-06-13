from tensorflow.keras import Layer
import tensorflow as tf


class PointNetLayer(Layer):
    """
    For each particle in the event, it evaluates the MLP and returns the same particle
    with a new set of features giveb by the output of the MLP.
    """

    def __init__(self, mlp: Layer, **kwargs):
        super().__init__(**kwargs)
        self._mlp = mlp
        self._mlp_output_dim = mlp.output_shape[-1]

    def compute_output_shape(self, input_shape):
        """For each sample the output shape is (number of particles, number of features from the MLP)"""
        batch_size = input_shape[0]
        number_of_particles = input_shape[1]
        return batch_size, number_of_particles, self._mlp_output_dim

    def call(self, events):
        """Evaluates the MLP in each particle in all events"""
        # number of events (batch size) and number of particles per event
        number_of_evts, number_of_particles_per_evt, number_of_particles_features = events.shape
        # we need to reshape in a way that we only have a list of particles
        all_particles = tf.reshape(events, (number_of_particles_per_evt * number_of_evts, number_of_particles_features))
        # now we need to apply the MLP in all the particles
        mlp_output = self._mlp(all_particles)
        # the last step is to reshape it again in the of (events, particles, output of the MLP)
        return tf.reshape(mlp_output, (number_of_evts, number_of_particles_per_evt, self._mlp_output_dim))


class EdgeConvLayer(tf.keras.layers.Layer):
    """
    Edge Convolution Layer for the Particle Cloud NN.
    The implementation strategy was taken from the paper https://arxiv.org/abs/1902.08570
    """

    def __init__(self, mlp: Layer, k_neighbors: int, coord_index, **kwargs):
        super().__init__(**kwargs)
        self._k_neighbors = k_neighbors
        self._coord_index = coord_index
        self._mlp = mlp
        self._mlp_output_dim = mlp.output_shape[-1]

    def call(self, events):
        """Evaluates the MLP in each edge of all particle clouds."""
        # tf.expand_dims, tf.tile
        # tf.repeat, tf.reshap
        pass

class ChannelWiseGlobalAveragePooling(Layer):

    def call(self, events):
        """Returns the feature-wise avarage over all the particles in the event"""
        return tf.reduce_mean(events, axis=1)

    def compute_output_shape(self, input_shape):
        batch_size, particles_features = input_shape[0], input_shape[-1]
        return batch_size, particles_features
