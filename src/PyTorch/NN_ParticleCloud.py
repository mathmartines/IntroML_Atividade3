from src.PyTorch.EdgeConvLayer import EdgeConvLayer
from torch import nn
import torch


class ParticleCloud(nn.Module):
    """Particle Cloud NN"""

    def __init__(self):
        super().__init__()
        # ModelFiles to be used in the EdgeConv Layer
        self._mlp = nn.Sequential(
            # one layer with 12 input parameters (6 features per particle)
            nn.Linear(in_features=12, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )
        # EdgeConv Layer
        self._first_edgeconv_layer = EdgeConvLayer(mlp=self._mlp, k_neighbors=3, mlp_output_shape=32,
                                                   coordinates_index=(1, 3))
        # Last layer of trainning
        self._last_layer = nn.Linear(in_features=32, out_features=2)
        # Dropout to avoid overfitting
        self._dropout = nn.Dropout(p=0.1)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output_edge_conv_layer = self._first_edgeconv_layer(x)
        # applied to aggregate all particles in the cloud
        global_avg_pooling = torch.mean(output_edge_conv_layer, dim=1)
        result_last_layer = self._last_layer(global_avg_pooling)
        # applying the softmax
        return self._softmax(self._dropout(result_last_layer))


