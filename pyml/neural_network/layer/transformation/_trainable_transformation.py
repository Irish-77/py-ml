"""Abstract class for every trainable layer to ensure that the methods
to handle operations regarding their parameters are implemented."""

from abc import ABC, abstractmethod

class _TrainableTransformation(ABC):
    """Abstract class describing which methods trainable layers need to implement
    regarding the replacement or exchange of parameters.
    """

    @abstractmethod
    def get_parameters(self):
        """Return parameters of this layer
        """
        pass

    @abstractmethod
    def set_parameters(self) -> None:
        """Set parameters
        """
        pass

