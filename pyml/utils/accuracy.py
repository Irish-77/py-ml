"""This module contains classes for calculating accuracy metrics in machine learning tasks.

It provides different accuracy calculation methods for classification and regression models.

Classes
-------
_Accuracy (ABC):
    Abstract base class defining the structure for accuracy metrics.
    Subclasses must implement methods for initialization and comparison.
    Also, methods are provided for calculating accuracy and handling accumulated accuracy.

MultiClassAccuracy:
    A subclass of _Accuracy specifically for multi-class classification models.
    Implements methods to compare predictions and ground truth values for accuracy calculation.

BinaryClassAccuracy:
    A subclass of _Accuracy designed for binary classification models.
    Compares predictions and ground truth values to compute accuracy.

RegressionAccuracy: 
    A subclass of _Accuracy tailored for regression models.
    Calculates accuracy based on precision and compares predictions to ground truth.
"""

from abc import ABC, abstractmethod
import numpy as np

class _Accuracy(ABC):
    """Abstract base class defining the structure for accuracy metrics.
    
    Subclasses must implement methods for initialization and comparison.
    Also, methods are provided for calculating accuracy and handling accumulated accuracy.

    Attributes
    ----------
    accumulated_sum : float
        Sum of matching values in accumulated accuracy calculation.
    accumulated_count : int
        Total count of samples for accumulated accuracy.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.reset()
    
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions:np.ndarray, y:np.ndarray) -> float:
        """Calculate accuracy based on predictions and ground truth values.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values.
        y : numpy.ndarray
            Ground truth values.

        Returns
        -------
        float
            Calculated accuracy.
        """
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self) -> float:
        """Calculate accumulated accuracy.

        Returns
        -------
        float
            Accumulated accuracy.
        """

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def reset(self) -> None:
        """Reset variables for accumulated accuracy.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


    @abstractmethod
    def init(self, y:np.ndarray) -> None:
        """Abstract method to initialize accuracy calculation parameters that must be overwritten.

        Parameters
        ----------
        y : numpy.ndarray
            Ground truth values.
        """
        pass

    @abstractmethod
    def compare(self, predictions:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Abstract method to compare predictions to ground truth values for accuracy calculation that must be overwritten.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values.
        y : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        pass

# Accuracy calculation for classification model
class MultiClassAccuracy(_Accuracy):
    """Subclass of _Accuracy specifically for multi-class classification models.
    
    Implements methods to compare predictions and ground truth values for accuracy calculation.
    """

    def __init__(self) -> None:
        """Initialize MultiClassAccuracy instance.
        """
        super().__init__()

    # No initialization is needed
    def init(self, y:np.ndarray) -> None:
        """No initialization needed for MultiClassAccuracy.

        Parameters
        ----------
        y : numpy.ndarray
            Ground truth values.
        """
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Compare predictions to ground truth values for multi-class accuracy.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values.
        y : numpy.ndarray
            Ground truth values.

        Returns
        -------
        numpy.ndarray
            Array of comparison results.
        """
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# Accuracy calculation for classification model
class BinaryClassAccuracy(_Accuracy):
    """Subclass of _Accuracy designed for binary classification models.

    Compares predictions and ground truth values to compute accuracy.
    """

    def __init__(self) -> None:
        """Initialize BinaryClassAccuracy instance.
        """
        super().__init__()

    # No initialization is needed
    def init(self, y:np.ndarray) -> None:
        """No initialization needed for BinaryClassAccuracy.

        Parameters
        ----------
        y : numpy.ndarray
            Ground truth values.
        """
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Compare predictions to ground truth values for binary class accuracy.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values.
        y : numpy.ndarray
            Ground truth values.

        Returns
        -------
        numpy.ndarray
            Array of comparison results.
        """
        return predictions == y

# Accuracy calculation for regression model
class RegressionAccuracy(_Accuracy):
    """Subclass of _Accuracy tailored for regression models.
    
    Calculates accuracy based on precision and compares predictions to ground truth.

    Attributes
    ----------
    precision : float
        Precision value used for regression accuracy calculation.
    """

    def __init__(self) -> None:
        """Initialize RegressionAccuracy instance.
        """
        super().__init__()
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed-in ground truth values
    def init(self, y:np.ndarray, reinit:bool=False) -> None:
        """Initialize regression accuracy calculation parameters.

        Parameters
        ----------
        y : numpy.ndarray
            Ground truth values.
        reinit : bool, optional
            Reinitialize precision if True, by default False
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Compare predictions to ground truth values for regression accuracy.

        Parameters
        ----------
        predictions : numpy.ndarray
            Predicted values.
        y : numpy.ndarray
            Ground truth values.

        Returns
        -------
        numpy.ndarray
            Array of comparison results.
        """
        return np.absolute(predictions - y) < self.precision