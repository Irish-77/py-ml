"""ReLU is an activation function used mainly in hidden layers
"""

from pyml.neural_network.layer.activation import _Activation
import numpy as np

class ReLU(_Activation):
    """Rectified linear unit (ReLU activation function)

    The ReLU function is defined as follows:
    :math:`f(x)=x^{+}=\max(0,x)={\\frac {x+|x|}{2}}={\\begin{cases}x&{\\text{if }}x>0, \\\ 0& {\\text{otherwise}}.\end{cases}}`
    """
   
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """Computes a forward pass

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous neural layer.
        """

        self.inputs = inputs

        # Computes ReLU activation
        self.output = np.maximum(0, inputs)

    
    def backward(self, dvalues:np.ndarray) -> None:
        """Computes the backward step

        The derivative of the softmax function will be calculated as follows:
        :math:`f'(x)={\\begin{cases}1&{\\text{if }}x>0,\\\ 0&{\\text{if }}x<0.\end{cases}}`.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived loss from the previous layers (reversed order).
        """

        self.dinputs = dvalues.copy()

        # Zero the if output of ReLu activation is smaller/equal than zero  
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs:np.ndarray) -> np.ndarray :
        """Converts outputs to predictions

        Returns the outputs computed by itself without any changes.
        However, in practice this activation function is rarely used as a final output function - neither for regression nor for classification.
        
        Parameters
        ----------
        outputs : np.ndarray
            Outputs computed by the final activation function

        Returns
        -------
        np.ndarray
            Returns same values as passed to this method
        """
        return outputs