"""Adagrad Optimizer for Neural Networks.
"""

from pyml.neural_network.layer.transformation import _Transformation
from pyml.neural_network.optimizer import _Optimizer

import numpy as np

class Adagrad(_Optimizer):
    """Adagrad Optimizer for Neural Networks.

    This optimizer adapts the learning rate of each parameter based on the
    historical gradient information. It uses squared gradients to scale the
    learning rate for each parameter separately.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate, by default 1.0
    decay : float, optional
        Learning rate decay factor, by default 0.0
    epsilon : float, optional
        Small value added to the denominator to prevent division by zero, by default 1e-7
    """

    def __init__(
        self,
        learning_rate:float=1.,
        decay:float=0.,
        epsilon:float=1e-7
    ) -> None:
        super().__init__(learning_rate, decay)
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon

    # Update parameters
    def update_parameters(self, layer:_Transformation) -> None:
        """Update the parameters of the given layer using the Adagrad optimization algorithm.

        Parameters
        ----------
        layer : _Transformation
            The layer to update.

        Note
        ----
        If the layer does not have cache arrays for weights and biases,
        this method initializes them with zeros. It then updates the cache
        with squared gradients and performs parameter updates using Adagrad.
        """

        # Check if the layer has cache arrays for weight and bias gradients.
        # If not, initialize them with zeros.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update the cache arrays with the squared current gradients.
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2


        # Perform a parameter update using Vanilla SGD,
        # and normalize the update using the square root of the cache arrays.
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
