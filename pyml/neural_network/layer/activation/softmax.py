"""Softmax activation function used for multiclass classification problems as
final compoment of the network.
"""

from pyml.neural_network.layer.activation import _Activation
import numpy as np

class Softmax(_Activation):
    """Softmax activation function

    The softmax activation function is defined as
    :math:`\sigma (\mathbf {z} )_{j}={\\frac {e^{z_{j}}}{\sum _{k=1}^{K}e^{z_{k}}}}`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs:np.ndarray) -> None:
        """Computes a forward pass

        The softmax activation function is defined as
        :math:`\sigma (\mathbf {z} )_{j}={\\frac {e^{z_{j}}}{\sum _{k=1}^{K}e^{z_{k}}}}`.

        Parameters
        ----------
        inputs : np.ndarray
            Input values from previous neural layer.
        """

        self.inputs = inputs

        # Compute unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Add noise to data
        # exp_values = exp_values + 0.000001 * np.random.randn(*exp_values.shape)

        # Normalize probabilities for each data point (axis=1)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues:np.ndarray) -> None:
        """Computes the backward step

        The derivative of the softmax function will be calculated as follows:
        :math:`\\frac {\partial S(z_{i})}{\partial z_{j}} = {\\begin{cases}  S(z_{i}) \cdot (1 - S(z_{i})) & \\text{if } i = j \\\ - S(z_{i}) \cdot S(z_{j}) & \\text{if } i \\neq j \end{cases}}`.
        
        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layers (reversed order).
        """

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


    def predictions(self, outputs:np.ndarray) -> np.ndarray:
        """Converts outputs to predictions

        The softmax activation function calculates the confidence for each class.
        Since the the softmax function does not only support binary tasks, but also multiclass problems,
        the user must receive the index of the class with the highest confidence score.

        Parameters
        ----------
        outputs : numpy.ndarray
            Output computed by the softmax activation function

        Returns
        -------
        numpy.ndarray
            Matrix containing the class indices with the highest confidence
        """
        return np.argmax(outputs, axis=1)