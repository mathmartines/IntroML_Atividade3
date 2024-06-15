import tensorflow as tf


class EdgeConvLayer(tf.keras.layers.Layer):
    """
    Edge Convolution Layer for the Particle Cloud NN.
    The implementation strategy was taken from the paper https://arxiv.org/abs/1902.08570
    """

    def __init__(self,
                 mlp,
                 k_neighbors: int, coord_index, **kwargs):
        super().__init__(**kwargs)
        self._k_neighbors = k_neighbors
        self._initial_coord, self._final_coord = coord_index
        self._mlp = mlp
        self._mlp_output_dim = mlp.output_shape[-1]

    def call(self, events):
        """Evaluates the ModelFiles in each edge of all particle clouds."""
        batch_size = tf.shape(events)[0]
        number_of_particles = tf.shape(events)[1]
        number_of_particles_features = tf.shape(events)[2]
        # getting the indices of each particle neighbor
        neighbors_indices = self.find_neighbors_indices(events)

        # for each particle and its neighbor, we need to compute the difference of features (the edge features)
        neighbors = self.gather_neighbors(events, neighbors_indices)
        central_particles = tf.repeat(events, repeats=self._k_neighbors, axis=1)
        central_particles = tf.reshape(central_particles,
                                       [batch_size, number_of_particles, self._k_neighbors,
                                        number_of_particles_features])
        edge_features = tf.concat([central_particles, neighbors - central_particles], axis=-1)
        # now we need to reshape it in a that's acceptable by the ModelFiles
        total_edges = batch_size * number_of_particles * self._k_neighbors
        edge_features = tf.reshape(edge_features, [total_edges, 2 * number_of_particles_features])
        # applying the ModelFiles to every edge
        mlp_output = self._mlp(edge_features)
        # reshape again into events as set of particles
        mlp_output = tf.reshape(mlp_output, [batch_size, number_of_particles, self._k_neighbors, self._mlp_output_dim])

        # return the mean value of each feature over the neighbors
        return tf.reduce_mean(mlp_output, axis=2)

    def gather_neighbors(self, events, neighbors_indices):
        """Returns the k-neighbors for each particle in an event"""
        number_of_evts = tf.shape(events)[0]
        number_of_particles = tf.shape(events)[1]

        # constructing the indices for each event
        event_indices = tf.range(number_of_evts)[:, tf.newaxis, tf.newaxis]
        event_indices = tf.tile(event_indices, [1, number_of_particles, self._k_neighbors])
        # combine the indices of each neighbor with the event indices
        combined_indices = tf.stack([event_indices, neighbors_indices], axis=-1)

        # return the neighbors (number of evts, number of particles, number of neighbors, number of features)
        return tf.gather_nd(events, combined_indices)

    def find_neighbors_indices(self, events):
        """Finds the indices of the k-closest neighbors"""
        batch_size = tf.shape(events)[0]
        number_of_particles_per_evt = tf.shape(events)[1]
        number_of_particles_features = tf.shape(events)[2]
        # First we need to calculate the closest k-neighbors
        # the events have dimensiont = (# evts, # particles, particle features)
        # creating a tensor with dim (# evts, # particles, 1, particle features)
        events_repeated_same_particle = tf.expand_dims(events, axis=2)
        # repeating the features of each particle by the number of particles in the event
        events_repeated_same_particle = tf.tile(events_repeated_same_particle, [1, 1, number_of_particles_per_evt, 1])
        # doing the same process, but the dim before last will have the features of each particle
        events_repeated_diff_particle = tf.repeat(events, repeats=number_of_particles_per_evt, axis=0)
        events_repeated_diff_particle = tf.reshape(events_repeated_diff_particle,
                                                   shape=[batch_size, number_of_particles_per_evt,
                                                          number_of_particles_per_evt, number_of_particles_features])
        # calculating the distance from the central particle and finding the indices of the k-closest particles
        diff = events_repeated_diff_particle - events_repeated_same_particle
        distances = tf.norm(diff[:, :, :, self._initial_coord: self._final_coord], axis=-1)
        # indices of the neighbors (excluding the own particle)
        return tf.argsort(distances, axis=-1)[:, :, 1: self._k_neighbors + 1]

    def compute_output_shape(self, input_shape):
        batch_size, number_of_particles = input_shape[0], input_shape[1]
        return batch_size, number_of_particles, self._mlp_output_dim

