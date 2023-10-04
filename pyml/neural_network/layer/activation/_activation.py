"""Abstract class for the activation functions ensuring the
implementation of the predictions method"""

from abc import ABC, abstractmethod
from pyml.neural_network.layer import _Layer

class _Activation(_Layer, ABC):
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predictions(self):
        """Converts the calculated output into actual predictions"""
        pass