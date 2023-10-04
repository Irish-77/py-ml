"""Linear activation function used for regression tasks as
final compoment of the network
"""

from pyml.neural_network.layer.activation import _Activation
import numpy as np

class Linear(_Activation):
    """Linear activation function

    The derivative of the linear activation function :math:`f(x) = x` is :math:`f'(x) = 1`.
    Hence, it's always one, which is why we can just pass through the given dvalues from the previous layer during the backpropagation step.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """Computes a forward pass

        The output of a linear layer is the input.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous neural layer.
        """

        self.inputs = inputs
        self.outputs = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        """Computes the backward step

        Since the derivative of a linear function is always one,
        we can keep the values.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layer (reversed order).
        """
        self.dinputs = dvalues.copy()


    def predictions(self, outputs:np.ndarray) -> np.ndarray:
        """Converts outputs to predictions

        Since this is a linear activation function,
        there is no need to convert any outputs.

        Parameters
        ----------
        outputs : numpy.ndarray
            Output computed by the linear activation function

        Returns
        -------
        numpy.ndarray
            Returns same values as passed to this method
        """
        return outputs
        