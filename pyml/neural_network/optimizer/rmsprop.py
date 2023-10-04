"""RMSProp Optimizer for Neural Networks.
"""

from pyml.neural_network.layer.transformation import _Transformation
from pyml.neural_network.optimizer import _Optimizer

import numpy as np

class RMSProp(_Optimizer):
    """RMSProp Optimizer for Neural Networks.

    This optimizer uses the Root Mean Square Propagation (RMSProp) algorithm
    to adapt the learning rate of each parameter based on the historical squared
    gradients.
    
    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate, by default 0.001
    decay : float, optional
        Learning rate decay factor, by default 0
    epsilon : float, optional
        Small value added to the denominator to prevent division by zero,
        by default 1e-7
    rho : float, optional
        Exponential moving average factor for squared gradients, by default 0.9
    """

    def __init__(
        self,
        learning_rate:float=0.001,
        decay:float=0.,
        epsilon:float=1e-7,
        rho:float=0.9
    ) -> None:
        super().__init__(learning_rate, decay)

        self.epsilon = epsilon
        self.rho = rho


    # Update parameters
    def update_parameters(self, layer:_Transformation) -> None:
        """Update the parameters of the given layer using the RMSProp optimization algorithm.

        Parameters
        ----------
        layer : _Transformation
            The layer to update.

        Note
        ----
        If the layer does not have cache arrays for weights and biases,
        this method initializes them with zeros. It then updates the cache
        with squared gradients using the RMSProp algorithm and performs parameter
        updates accordingly.
        """

        # Check if the layer has cache arrays for weight and bias gradients.
        # If not, initialize them with zeros.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update the cache arrays with the squared current gradients.
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2


        # Perform a parameter update using the Vanilla Stochastic Gradient Descent (SGD) algorithm,
        # and normalize the update using the square root of the cache arrays.
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)