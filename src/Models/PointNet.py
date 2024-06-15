import keras
from src.Layers.PointNetLayer import PointNetLayer
from src.Layers.Pooling import ChannelWiseGlobalAveragePooling


class PointNet(keras.Model):
    """Neural network architecture for the PointNet model"""

    def __init__(self, mlp, **kwargs):
        super().__init__(**kwargs)
        # Multi-Layer Perceptron that will be applied to all the particles in the event
        self.point_net = PointNetLayer(mlp)
        self.channel_wise_pooling = ChannelWiseGlobalAveragePooling()
        # Dense Layers after the PointNet layers
        self.layer = keras.layers.Dense(64)
        self.dropout = keras.layers.Dropout(0.1)
        self.relu = keras.layers.ReLU()

    def call(self, inputs):
        """Applying the PointNet model"""
        # Applying the PointNet
        output_point_net = self.point_net(inputs)
        # Channel wise avarage pooling
        output_channel_wise = self.channel_wise_pooling(output_point_net)
        # dense layers
        output_layer_1 = self.layer(output_channel_wise)
        output_layer_1 = self.dropout(output_layer_1)
        return self.relu(output_layer_1)
