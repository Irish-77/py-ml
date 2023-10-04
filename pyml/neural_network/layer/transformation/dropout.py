"""Dropout layer used during training to avoid overfitting
"""

from pyml.neural_network.layer.transformation import _Transformation
import numpy as np


class Dropout(_Transformation):
    """Dropout Layer for Neural Networks.

    This layer applies dropout regularization during training to prevent
    overfitting by randomly setting a fraction of the input units to zero
    during each forward pass.

    Parameters
    ----------
    rate : float
        The dropout rate, indicating the fraction of input units to drop.
        The remaining units will be scaled by 1 / (1 - rate).

    Attributes
    ----------
    generator :numpy.random.Generator
        Random number generator used for mask generation.
    inputs : numpy.ndarray
        The input values received during the forward pass.
    binary_mask : numpy.ndarray
        Binary mask with probabilities of keeping units.
    output : numpy.ndarray
        The output values after applying dropout during the forward pass.
    """

    def __init__(self, rate:float) -> None:

        super().__init__()
        
        # User specifies the rate of dropout -> Convert to rate of keeping the nodes
        self.rate = 1 - rate

        self.generator = np.random.default_rng()

    def forward(self, inputs:np.ndarray, training:bool) -> None:
        """Perform the forward pass with dropout during training and without dropout during inference.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input values from previous layer.
        training : bool
            If true, apply dropout to layer be replacing some inputs with zero.
            Otherwise inputs will be passed through as outputs

        Note
        ----
        During training, the layer generates a binary mask with probabilities of
        keeping units, which is then applied to the inputs to create the output.
        During inference (when training is False), the layer simply copies the inputs
        to the output.
        """
        self.inputs = inputs
        
        # TODO optimize training parameter
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate masks of layer with probabilites of keeping the 
        self.binary_mask = self.generator.binomial(1,self.rate,size=inputs.shape)

        # Apply mask and set some outputs to zero
        self.output = inputs * self.binary_mask

    def backward(self, dvalues:np.ndarray) -> None:
        """Perform the backward pass by applying the binary mask to gradients.

        Parameters
        ----------
        dvalues : np.ndarray
            Derived gradient from the previous layer (reversed order).
        """
        self.dinputs = dvalues * self.binary_mask
