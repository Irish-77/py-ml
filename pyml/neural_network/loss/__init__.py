"""Loss Functions Module

This package contains various loss functions commonly used in neural network training.
Each loss function is implemented as a separate module within the package.
"""

from ._loss import _Loss
from .categorical_cross_entropy import CategoricalCrossentropy
from .binary_cross_entropy import BinaryCrossentropy
from .mean_squarred_error import MeanSquaredError
from .mean_absolute_error import MeanAbsoluteError
from .softmax_loss_categorical_cross_entropy import Softmax_CategoricalCrossentropy