"""Base (parent) class describing the abstract design for the component parts of a neural network
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class _Layer(ABC):
    """Abstract class describing the layers of a network (e.g. activation functions, dense, dropout).

    Attributes
    ----------  
    previous_layer : _Layer
        Layer that is previous to this layer.
    next_layer : _Layer
        Layer that is subsequent to this layer.
    """

    def __init__(self):
        self.previous_layer = None
        self.next_layer = None

    def set_adjacent_layers(self, previous_layer:_Layer, next_layer:_Layer):
        """Set adjacent layers which are needed for the model to iterate through the layers.

        Parameters
        ----------
        previous_layer : _Layer
            Layer that is previous to this layer.
        next_layer : _Layer
            Layer that is subsequent to this layer.
        """
        self.previous_layer = previous_layer
        self.next_layer = next_layer

    @abstractmethod
    def forward(self, inputs:np.ndarray):
        """Abstract forward pass
        
        Inputs must be saved as attribute of the class,
        since they will be accessed during the backward pass.
        """

        #self.inputs = inputs
        
        pass

    @abstractmethod
    def backward(self):
        """Abstract backward step
        """
        pass