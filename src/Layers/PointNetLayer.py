import tensorflow as tf

class PointNetLayer(tf.keras.layers.Layer):
    """
    For each particle in the event, it evaluates the ModelFiles and returns the same particle
    with a new set of features giveb by the output of the ModelFiles.
    """

    def __init__(self, mlp, **kwargs):
        super().__init__(**kwargs)
        self._mlp = mlp
        self._mlp_output_dim = mlp.output_shape[-1]

    def compute_output_shape(self, input_shape):
        """For each sample the output shape is (number of particles, number of features from the ModelFiles)"""
        batch_size = input_shape[0]
        number_of_particles = input_shape[1]
        return batch_size, number_of_particles, self._mlp_output_dim

    def call(self, events):
        """Evaluates the ModelFiles in each particle in all events"""
        # number of events (batch size) and number of particles per event
        number_of_evts = tf.shape(events)[0]
        number_of_particles_per_evt = tf.shape(events)[1]
        number_of_particles_features = tf.shape(events)[2]
        # we need to reshape in a way that we only have a list of particles
        all_particles = tf.reshape(events, (number_of_particles_per_evt * number_of_evts, number_of_particles_features))

        # now we need to apply the ModelFiles in all the particles
        mlp_output = self._mlp(all_particles)
        # the last step is to reshape it again in the of (events, particles, output of the ModelFiles)
        return tf.reshape(mlp_output, (number_of_evts, number_of_particles_per_evt, self._mlp_output_dim))
