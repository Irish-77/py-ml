"""Module defining the Softmax Loss Categorical Cross-Entropy class for combined softmax activation and categorical cross-entropy loss.

This module contains the implementation of the Activation_Softmax_Loss_CategoricalCrossentropy class,
which represents the combination of the softmax activation function and the categorical cross-entropy loss.
It provides a method for the backward pass computation of the gradient with respect to the predicted values.
"""

import numpy as np

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Softmax_CategoricalCrossentropy():
    """Combined Softmax Activation and Categorical Cross-Entropy loss.

    This class defines the backward pass computation for calculating the gradient of the loss
    with respect to the predicted values when using the Softmax activation and Categorical Cross-Entropy loss.

    Attributes
    ----------
    dinputs : numpy.ndarray
        Gradient of the loss with respect to the predicted values.
    """

    # Backward pass
    def backward(self, dvalues:np.ndarray, y_true:np.ndarray) -> None:
        """Compute the backward pass to calculate the gradient of the loss with respect to the predicted values.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Gradient of the loss with respect to the predicted values.
        y_true : numpy.ndarray
            Ground truth values.
        """

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples