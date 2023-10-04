"""Input layer used by the NN class when creating the model instance
"""

from pyml.neural_network.layer.transformation import _Transformation
import numpy as np

class Input(_Transformation):
    """Input layer used for the neural network
    """

    def __init__(self) -> None:
        
        super().__init__()

    # Not needed but must be overwritten
    def backward(self) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> None:
        """Forward pass

        Parameters
        ----------
        inputs : np.ndarray
            Handles the input of the user
        """

        self.output = inputs
