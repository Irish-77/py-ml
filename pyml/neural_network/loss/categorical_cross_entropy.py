"""Module defining the Categorical Cross-Entropy class for multi-class classification tasks.

This module contains the implementation of the CategoricalCrossentropy class,
which represents a loss function for multi-class classification problems. It provides
methods for both the forward and backward pass computations of the Categorical Cross-Entropy loss.
"""

from pyml.neural_network.loss import _Loss
import numpy as np

class CategoricalCrossentropy(_Loss):
    """Categorical Cross-Entropy loss function for multi-class classification tasks.

    This class defines the forward and backward pass computations for calculating
    the Categorical Cross-Entropy loss and its gradient with respect to the predicted values.

    Attributes
    ----------
    dinputs : numpy.ndarray
        Gradient of the loss with respect to the predicted values.
    """

    # Forward pass
    def forward(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        """Compute the forward pass of the Categorical Cross-Entropy loss.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicted output from the model, representing class probabilities.
        y_true : numpy.ndarray
            Ground truth values, containing integer class labels or one-hot encoded vectors.


        Returns
        -------
        numpy.ndarray
            Array of negative log-likelihoods for each sample.
        """

        # Number of samples in a batch
        samples = len(y_pred)


        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues:np.ndarray, y_true:np.ndarray) -> None:
        """Compute the backward pass to calculate the gradient of the loss with respect to the predicted values.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Gradient of the loss with respect to the predicted values.
        y_true : numpy.ndarray
            Ground truth values, containing integer class labels or one-hot encoded vectors.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples