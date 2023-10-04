"""
Module defining the Binary Cross-Entropy class for binary classification tasks.

This module contains the implementation of the BinaryCrossentropy class,
which represents a loss function for binary classification problems. It provides
methods for both the forward and backward pass computations of the Binary Cross-Entropy loss.
"""

from pyml.neural_network.loss import _Loss
import numpy as np

# Binary cross-entropy loss
class BinaryCrossentropy(_Loss):
    """Binary Cross-Entropy loss function for binary classification tasks.

    This class defines the forward and backward pass computations for calculating
    the Binary Cross-Entropy loss and its gradient with respect to the predicted values.

    Attributes
    ----------
    dinputs : numpy.ndarray
        Gradient of the loss with respect to the predicted values.
    """

    # Forward pass
    def forward(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        """Compute the forward pass of the Binary Cross-Entropy loss.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicted output from the model, representing class probabilities.
        y_true : numpy.ndarray
            Ground truth values, containing binary labels.

        Returns
        -------
        numpy.ndarray
            Array of sample-wise binary cross-entropy losses.
        """

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

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
            Ground truth values, containing binary labels.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])


        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples