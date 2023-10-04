"""Abstract class for the optimizers"""

from abc import ABC, abstractmethod
from pyml.neural_network.layer.transformation import _Transformation
from pyml.exceptions import OutsideSpecifiedRange

class _Optimizer(ABC):
    """Abstract class for neural network optimizers.

    This abstract class defines the basic structure and behavior of optimizers
    for neural network training. Concrete optimizer classes should inherit from
    this class and provide implementations for the required methods.

    Parameters
    ----------
    learning_rate : float, optional 
        The initial learning rate, by default 1.0
    decay : float, optional
        Learning rate decay factor, dy default 0.0

    Attributes
    ----------
    current_learning_rate : float
        The current effective learning rate, considering decay and iterations.
    """

    def __init__(self, learning_rate:float=1., decay:float=0.) -> None:
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        if decay < 0 or decay > 1:
            raise OutsideSpecifiedRange(decay, 'Decay', 0, 1)
        self.decay = decay

        # Set iterations to zero
        self.iterations = 0

    def pre_update_parameters(self) -> None:
        """Update the current learning rate based on decay.

        This method calculates and updates the current learning rate based on the
        decay factor and the number of iterations performed.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    @abstractmethod
    def update_parameters(self, layer:_Transformation) -> None:
        """
        Abstract method to update the parameters of a layer using the optimizer.

        Concrete optimizer classes must implement this method to update the
        parameters of a given layer using the specific optimization algorithm.

        Parameters
        ----------
        layer : _Transformation
            The layer to update.
        """
        pass

    def post_update_parameters(self) -> None:
        """Updates the iteration counter after each layer update
        """
        self.iterations += 1