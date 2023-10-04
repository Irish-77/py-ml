"""Module defining the Mean Squared Error class for mean squared error loss.

This module contains the implementation of the Loss_MeanSquaredError class,
which represents the mean squared error loss function. It provides methods for both
the forward and backward pass computations of the mean squared error loss.
"""

from pyml.neural_network.loss import _Loss
import numpy as np


# Mean Absolute Error loss
class MeanAbsoluteError(_Loss):  # L1 loss
    """Mean Squared Error loss function.

    This class defines the forward and backward pass computations for calculating
    the mean squared error loss and its gradient with respect to the predicted values.

    Attributes
    ----------
    dinputs : numpy.ndarray
        Gradient of the loss with respect to the predicted values.
    """

    def forward(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        """Compute the forward pass of the mean squared error loss.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicted output from the model.
        y_true : numpy.ndarray
            Ground truth values.

        Returns
        -------
        numpy.ndarray
            Array of sample-wise mean squared error losses.
        """

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

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
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
