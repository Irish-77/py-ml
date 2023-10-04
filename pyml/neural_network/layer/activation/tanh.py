"""Tanh activation function is mainly used for classification between two classes"""

from pyml.neural_network.layer.activation import _Activation
import numpy as np

class Tanh(_Activation):
    """Tanh activation function

    The tanh function is defined as:

    :math:`\\tanh x = {\\frac {\sinh x}{\cosh x}} = {\\frac {\mathrm {e} ^{x}-\mathrm {e} ^{-x}}{\mathrm {e} ^{x}+\mathrm {e} ^{-x}}} = {\\frac {\mathrm {e} ^{2x}-1}{\mathrm {e} ^{2x}+1}}=1-{\\frac {2}{\mathrm {e} ^{2x}+1}}`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs:np.ndarray) -> None:
        """Computes a forward pass

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous neural layer.
        """
        self.inputs = inputs

        self.output = np.tanh(inputs)

    def backward(self, dvalues:np.ndarray) -> None:
        """Computes the backward step

        The derivative of the tanh function will be calculated as follows:

        :math:`\dfrac{d\\tanh}{dx}=1-\\tanh^2=\dfrac{1}{\cosh^2 x}`.
        
        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layers (reversed order).
        """
        return super().backward()