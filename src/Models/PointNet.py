import keras
from src.Layers.PointNetLayer import PointNetLayer
from src.Layers.Pooling import ChannelWiseGlobalAveragePooling


class PointNet(keras.Model):
    """Neural network architecture for the PointNet model"""

    def __init__(self, mlp, mlp_output_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp = mlp
        self.mlp_output_dim = mlp_output_dim
        # Multi-Layer Perceptron that will be applied to all the particles in the event
        self.point_net = PointNetLayer(self.mlp, self.mlp_output_dim)
        self.channel_wise_pooling = ChannelWiseGlobalAveragePooling()
        # Dense Layers after the PointNet layers
        self.layer = keras.layers.Dense(64)
        self.dropout = keras.layers.Dropout(0.1)
        self.relu = keras.layers.ReLU()
        self.layer2 = keras.layers.Dense(64)
        self.dropout2 = keras.layers.Dropout(0.1)
        self.relu2 = keras.layers.ReLU()

    def call(self, inputs):
        """Applying the PointNet model"""
        # Applying the PointNet
        output_point_net = self.point_net(inputs)
        # Channel wise avarage pooling
        output_channel_wise = self.channel_wise_pooling(output_point_net)
        # dense layers
        output_layer_1 = self.layer(output_channel_wise)
        output_layer_1 = self.dropout(output_layer_1)
        output_layer_1 = self.relu(output_layer_1)
        output_layer_2 = self.layer2(output_layer_1)
        output_layer_2 = self.dropout2(output_layer_2)

        return self.relu2(output_layer_2)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "mlp": keras.saving.serialize_keras_object(self.mlp),
            "mlp_output_dim": self.mlp_output_dim
        }
        return {**config, **base_config}

    @classmethod
    def from_config(cls, config):
        mlp = config.pop("mlp")
        mlp = keras.saving.deserialize_keras_object(mlp)
        return cls(mlp, **config)
