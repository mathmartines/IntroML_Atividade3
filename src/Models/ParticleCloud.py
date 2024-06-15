import keras
from src.Layers.EdgeConvLayer import EdgeConvLayer
from src.Layers.Pooling import ChannelWiseGlobalAveragePooling


class ParticleCloud(keras.Model):
    """Neural network architecture for the ParticleCloud model"""

    def __init__(self, mlp, mlp_output_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp = mlp
        self.mlp_output_dim = mlp_output_dim
        # Multi-Layer Perceptron that will be applied to all the particles in the event
        self.edge_conv_layer = EdgeConvLayer(mlp, mlp_output_dim, 3, (1, 3))
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
