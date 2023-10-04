"""Adam Optimizer for Neural Networks.
"""

from pyml.neural_network.layer.transformation import _Transformation
from pyml.neural_network.optimizer import _Optimizer

import numpy as np

class Adam(_Optimizer):
    """Adam Optimizer for Neural Networks

    This optimizer uses the Adam (Adaptive Moment Estimation) algorithm to
    adapt the learning rate of each parameter based on both the first and second
    moments of the gradients.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate, by default 0.001
    decay : float, optional
        Learning rate decay factor, by default 0.
    epsilon : float, optional
        Small value added to the denominator to prevent division by zero,
        by default 1e-7
    beta_1 : float, optional
        Exponential moving average factor for the first moment (mean) of gradients,
        by default 0.9
    beta_2 : float, optional
        Exponential moving average factor for the second moment (uncentered variance) of gradients,
        by default 0.999
    """

    def __init__(
        self,
        learning_rate:float=0.001,
        decay:float=0.,
        epsilon:float=1e-7,
        beta_1:float=0.9,
        beta_2:float=0.999
    ) -> None:        
        super().__init__(learning_rate, decay)

        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_parameters(self, layer:_Transformation) -> None:
        """Update the parameters of the given layer using the Adam optimization algorithm.

        Parameters
        ----------
        layer : _Transformation
            The layer to update.

        Note
        ----
        If the layer does not have cache arrays for weights and biases,
        this method initializes them with zeros. It then updates the momentums
        and caches of gradients using the Adam algorithm and performs parameter
        updates accordingly.    
        """

        # Check if the layer has cache arrays for weight and bias updates.
        # If not, initialize them with zeros.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)


        # Update the momentum arrays with the current gradients.
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        
        # Calculate the corrected momentum values.
        # self.iteration is 0 at the first pass, so we start with 1 here.
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update the cache arrays with squared current gradients.
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        
        # Calculate the corrected cache values.
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Perform a parameter update using Vanilla SGD,
        # and normalize the update using the square root of the cache arrays.
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)