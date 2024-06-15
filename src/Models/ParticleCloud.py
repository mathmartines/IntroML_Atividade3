import keras
from src.Layers.EdgeConvLayer import EdgeConvLayer
from src.Layers.Pooling import ChannelWiseGlobalAveragePooling


class ParticleCloud(keras.Model):
    """Neural network architecture for the ParticleCloud model"""

    def __init__(self, mlp, **kwargs):
        super().__init__(**kwargs)
        # Multi-Layer Perceptron that will be applied to all the particles in the event
        self.edge_conv_layer = EdgeConvLayer(mlp, 3, (1, 3))
        self.channel_wise_pooling = ChannelWiseGlobalAveragePooling()
        # Dense Layers after the PointNet layers
        self.layer_1 = keras.layers.Dense(64)
        self.dropout_1 = keras.layers.Dropout(0.1)
        self.relu_1 = keras.layers.ReLU()
        self.layer_2 = keras.layers.Dense(64)
        self.dropout_2 = keras.layers.Dropout(0.1)
        self.relu_2 = keras.layers.ReLU()

    def call(self, inputs):
        """Applying the PointNet model"""
        # Applying the EdgeConvLayer
        output_point_net = self.edge_conv_layer(inputs)
        # Channel wise avarage pooling
        output_channel_wise = self.channel_wise_pooling(output_point_net)
        # dense layers
        output_layer_1 = self.layer_1(output_channel_wise)
        output_layer_1 = self.dropout_1(output_layer_1)
        output_layer_1 = self.relu_1(output_layer_1)
        output_layer_2 = self.layer_2(output_layer_1)
        output_layer_2 = self.dropout_2(output_layer_2)
        return self.relu_2(output_layer_2)
