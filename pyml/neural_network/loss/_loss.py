
from abc import ABC, abstractmethod
from pyml.neural_network.layer import _Layer
import numpy as np


# Common loss class
class _Loss(ABC):
    """Abstract base class for defining loss functions in neural networks.

    This class provides methods for calculating data loss, regularization loss, and
    their combinations. It also handles accumulation of losses during training passes.

    Attributes
    ----------
    trainable_layers : list[_Layer])
        List of trainable layers in the model.
    accumulated_sum : float
        Accumulated sum of losses during training.
    accumulated_count : int
        Accumulated count of samples during training.
    """

    # Regularization loss calculation
    def regularization_loss(self) -> float:
        """Calculate the regularization loss based on L1 and L2 regularization terms
        of the trainable layers' weights and biases.

        Returns
        -------
        float
            Regularization loss value.
        """

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * \
                                              layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * \
                                              layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def set_trainable_layers(self, trainable_layers:list[_Layer]) -> None:
        """Set the list of trainable layers for loss calculations.

        Parameters
        ----------
        trainable_layers : list[_Layer]
            List of trainable layers in the model.
        """
        self.trainable_layers = trainable_layers


    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(
        self,
        output:np.ndarray,
        y:np.ndarray,
        *,
        include_regularization:bool=False
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """ Calculate the total loss based on model output and ground truth values.

        Parameters
        ----------
        output : numpy.ndarray
            Predicted output from the model.
        y : numpy.ndarray
            Ground truth values.
        include_regularization : bool, optional
            Whether to include regularization loss, by default False.

        Returns
        -------
        numpy.ndarray or tuple
            Total loss value if `include_regularization` is False, or a tuple of 
            data and regularization loss values if `include_regularization` is True.
        """

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(
        self,
        *,
        include_regularization:bool=False
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """Calculate the accumulated loss over a training pass.

        Parameters
        ----------
        include_regularization : bool, optional
            Whether to include regularization loss, by default False

        Returns
        -------
        float or tuple
            Accumulated loss value or tuple of data and regularization losses.
        """

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def reset(self) -> None:
        """Reset the accumulated loss variables for a new training pass.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @abstractmethod
    def forward(self, y_pred:np.ndarray, y_true:np.ndarray) -> None:
        """Abstract method for the forward pass of the loss function.

        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicted output from the model.
        y_true : numpy.ndarray
            Ground truth values.
        """
        pass

    @abstractmethod
    def backward(self, dvalues:np.ndarray, y_true:np.ndarray) -> None:
        """Abstract method for the backward pass of the loss function.

        Parameters
        ----------
        dvalues : numpy.ndarray
            Gradient of loss with respect to the predicted values.
        y_true : numpy.ndarray
            Ground truth values.
        """
        pass