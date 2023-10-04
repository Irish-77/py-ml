"""Stochastic Gradient Descent (SGD) Optimizer
"""

from pyml.neural_network.layer.transformation import _Transformation
from pyml.neural_network.optimizer import _Optimizer
from pyml.exceptions import OutsideSpecifiedRange

import numpy as np

class SGD(_Optimizer):
    """Stochastic Gradient Descent (SGD) Optimizer.

    This optimizer performs stochastic gradient descent with optional momentum
    and learning rate decay.
    
    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate, by default 1
    decay : float, optional
        The learning rate decay factor, by default 0
    momentum : float, optional
        The momentum factor for gradient updates, by default 0

    Raises
    ------
    OutsideSpecifiedRange
        If momentum value is outside the range [0, 1].
    """

    def __init__(
        self,
        learning_rate:float = 1,
        decay:float = 0,
        momentum:float=0
    ) -> None:

        super().__init__(learning_rate, decay)

        if momentum < 0 or momentum > 1:
            raise OutsideSpecifiedRange(momentum, 'Momentum', 0, 1)
        self.momentum = momentum
            
    def update_parameters(self, layer:_Transformation) -> None:
        """Update the weights and biases of the given layer using SGD.

        This method updates the weights and biases of the specified layer using
        stochastic gradient descent with optional momentum.

        Parameters
        ----------
        layer : _Transformation
            The layer to update.

        Note
        ----
        If the layer does not have momentum arrays for weights and biases,
        this method initializes them and performs updates using momentum.
        Otherwise, updates are performed without momentum.
        """

        # Check if the layer has momentum arrays for weight and bias updates.
        if not hasattr(layer, 'weight_momentums'):
            # If not, initialize them with zeros.
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

            # Compute momentums for weights and biases.
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (if momentum isn't used)
        else:
            # Calculate weight and bias updates without momentum.
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update the weights and biases using the computed updates.
        layer.weights += weight_updates
        layer.biases += bias_updates