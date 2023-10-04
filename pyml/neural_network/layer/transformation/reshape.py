"""Reshape Layer

This class represents a layer that reshapes the input data to a specified output shape.
It is commonly used to flatten or reshape data before passing it to other layers in a neural network.
"""

from pyml.neural_network.layer.transformation import _Transformation
import numpy as np

class Reshape(_Transformation):
    """Reshape Layer
    
    Parameters
    ----------
    input_shape : tuple[int]
        The shape of the input data before reshaping.
    output_shape : tuple[int]
        The desired shape of the output data after reshaping.
    
    Notes
    -----
    This layer does not have any learnable parameters.
    It simply reshapes the input data based on the specified input and output shapes.
    """

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs:np.ndarray) -> None:
        """Perform the forward pass of the reshaping operation on the input data.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous layer.
        """
        self.inputs = inputs

        self.output = np.reshape(inputs, self.output_shape)

    def backward(self, dvalues:np.ndarray) -> None:
        """Perform the backward pass of the reshaping operation on the gradient of the output data.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layer (reversed order).
        """
        
        # Save reshaped gradient
        self.dinputs = np.reshape(dvalues, self.input_shape)



class Flatten(_Transformation):
    """Flatten Layer
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs:np.ndarray) -> None:
        """Perform the forward pass of the flattening operation on the input data.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous layer.
        """
        self.inputs = inputs

        self.BATCH_SIZE = inputs.shape[0]
        self.input_channels = inputs.shape[1]
        self.height = inputs.shape[2]
        self.width = inputs.shape[3]
        

        self.output = np.reshape(inputs, (self.BATCH_SIZE, self.input_channels * self.width * self.height))

    def backward(self, dvalues:np.ndarray) -> None:
        """Perform the backward pass of the flattening operation on the gradient of the output data.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layer (reversed order).
        """
        
        # Save reshaped gradient
        self.dinputs = np.reshape(dvalues, (self.BATCH_SIZE, self.input_channels, self.height, self.width))