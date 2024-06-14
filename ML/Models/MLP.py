import keras


class MLP(keras.Model):
    """
    Simple Multi-Layer Perceptron

    The model has three hidden layers with the number of neurons given by the parameter num_neurons.
    Each hidden layer consists of a linear transformation followed by a BatchNormalization and
    ReLU activation function.
    """

    def __init__(self, num_neurons, **kwargs):
        super().__init__(**kwargs)
        # first layer
        self.first_transf = keras.layers.Dense(num_neurons)
        self.first_batch_norm = keras.layers.BatchNormalization()
        self.first_relu = keras.layers.ReLU()
        # second layer
        self.second_transf = keras.layers.Dense(num_neurons)
        self.second_batch_norm = keras.layers.BatchNormalization()
        self.second_relu = keras.layers.ReLU()
        # third layer
        self.third_transf = keras.layers.Dense(num_neurons)
        self.third_batch_norm = keras.layers.BatchNormalization()
        self.third_relu = keras.layers.ReLU()

    def call(self, inputs):
        """Evaluates the MLP in the set of inputs"""
        # Layer 1
        output_first_layer = self.first_transf(inputs)
        output_first_layer = self.first_batch_norm(output_first_layer)
        output_first_layer = self.first_relu(output_first_layer)
        # Layer 2
        output_second_layer = self.second_transf(output_first_layer)
        output_second_layer = self.second_batch_norm(output_second_layer)
        output_second_layer = self.second_relu(output_second_layer)
        # Layer 3
        output_third_layer = self.third_transf(output_second_layer)
        output_third_layer = self.third_batch_norm(output_third_layer)

        return self.third_relu(output_third_layer)
