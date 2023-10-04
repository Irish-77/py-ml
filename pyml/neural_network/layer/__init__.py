"""The layer include all parts of a neural network:
the actual layer consisting of the neurons (nodes and their connection), and the activation function, which processes the output of the neurons.
The term layer is in this package the abstract term for activation function and transformation layers (e.g. dense, dropout etc.).

layer:
    1. transformation
        1. dense
        2. dropout
        3. convolutional
    2. activation
        1. linear
        2. sigmoid
        3. softmax
        4. relu
"""

from ._layer import _Layer
from .transformation import *
from .activation import *