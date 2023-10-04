"""Sigmoid activation function used for binary classification problems as
final compoment of the network; alias for logistic function.
"""

from pyml.neural_network.layer.activation import _Activation
import numpy as np

class Sigmoid(_Activation):
    """Sigmoid activation function
    
    The sigmoid function :math:`\\sigma (x)={\\frac {1}{1+e^{-x}}}` 
    is a non-linear activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """Computes a forward pass

        Computes the confidences for each input using this function:
        :math:`\\sigma (x)={\\frac {1}{1+e^{-x}}}={\\frac {e^{x}}{e^{x}+1}}=1-\\sigma (-x)`.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous neural layer.
        """
        self.inputs = inputs
        
        # Compute sigmoid function on inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues:np.ndarray) -> None:
        """Computes the backward step

        The derivative of the sigmoid function is :math:`\\sigma' (x) = \\sigma (x) (1 - \\sigma (x))`.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layers (reversed order).
        """
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs:np.ndarray) -> np.ndarray:
        """Converts outputs to predictions

        Decodes the confidences for each prediction to binary predictions, meaning 0 or 1.  
        If single confidence is > 0.5, than 1, true etc. is set for prediction outcome.
        
        TODO check type of outputs, could also be np.array

        Parameters
        ----------
        outputs : numpy.ndarray
            Output computed by the sigmoid activation function

        Returns
        -------
        numpy.ndarray
            Matrix containing the class predictions; values are either zero or one
        """
        return (outputs > 0.5) * 1