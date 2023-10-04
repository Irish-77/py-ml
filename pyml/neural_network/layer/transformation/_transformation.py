"""Abstract class for the every component in a neural network that envolves connecting layers with each other"""

from abc import ABC, abstractmethod
from pyml.neural_network.layer import _Layer

class _Transformation(_Layer, ABC):
    """Abstract class describing the layers that connect the neurons.
    """
    def __init__(self) -> None:
        super().__init__()