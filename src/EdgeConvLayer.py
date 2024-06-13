import torch


class EdgeConvLayer(torch.nn.Module):
    """Edge convulation layer for the ParticleCloud NN"""

    def __init__(self, mlp, k_neighbors, mlp_output_shape, coordinates_index):
        super().__init__()
        # multi-layer perceptron that will be used as the internal
        # function that handles the edges
        self._mlp = mlp
        self._k_neighbors = k_neighbors
        self._mlp_output_shape = mlp_output_shape
        self._initial_coord, self._final_coord = coordinates_index

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        The EdgeConvLayer applies the MLP for each edge in the ParticleCloud,
        and takes a channel-wise avarage over the outputs of the MLP. The
        avarage over all the edges represents the new set of features for the particle.
        An edge represents the particle and its neighbor features. The number of edges is equal
        to the number of neighbors (k-neighbors).

        :param events: Tensor with the sets of particles
        :return:
        """
        return torch.stack([self.create_cloud(event) for event in events])

    def create_cloud(self, event: torch.Tensor) -> torch.Tensor:
        particles_number = len(event)
        output_particles = torch.empty(size=(particles_number, self._mlp_output_shape), dtype=torch.float32)

        # particle cloud for the current particle
        # it will store the output of the MLP for each edge
        particle_cloud = torch.empty(size=(self._k_neighbors, self._mlp_output_shape), dtype=torch.float32)

        # for each particle in the event we must find the particle cloud
        # that is, its k-closest neighbors
        for particle_index in range(particles_number):
            for index_cloud, neighbor_index in enumerate(self._find_neighbors(particle_index, event)):
                # setting up the right features for the MLP
                particle, neighbor = event[particle_index], event[neighbor_index]
                edge_features = torch.concat([particle, neighbor - particle], dim=0)
                particle_cloud[index_cloud] = self._mlp(edge_features)
            # the last step is to take the channel-wise avarage over the edges
            output_particles[particle_index] = torch.mean(particle_cloud, dim=0)

        return output_particles

    def _find_neighbors(self, particle_index: int, event: torch):
        """
        Finds the k neighbors of the particle at index particle_index.
        It returns the indices of the neighbors in the event.
        """
        # evaluating the distance from the particle
        distance_from_particle = torch.linalg.norm(
            event[:, self._initial_coord: self._final_coord] -
            event[particle_index, self._initial_coord: self._final_coord],
            dim=1
        )
        # getting the indices of the k closest particles and excluding the own particle
        return torch.argsort(distance_from_particle)[1:1 + self._k_neighbors]



