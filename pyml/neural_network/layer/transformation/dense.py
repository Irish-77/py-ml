"""Dense layer
"""

from pyml.neural_network.layer.transformation import _Transformation
from pyml.neural_network.layer.transformation import _TrainableTransformation
import numpy as np

class Dense(_Transformation, _TrainableTransformation):
    """Dense (Fully Connected) Layer for Neural Networks.

    This layer represents a fully connected (dense) layer in a neural network,
    where each input neuron is connected to each output neuron. The layer supports
    L1 and L2 weight and bias regularization.

    The output for the j-th is computed as follows 
    :math:`{\\text{output}}_{j}=\sum \limits _{i=1}^{n}x_{i}w_{ij}`.

    The dense layer allows to apply regularization, in particular L1 and L2.
    Both are regularization techniques aiming to prevent overfitting.
    L1 Regularization, also called Lasso Regression (in context of regressions),
    shrinks paramaters towards zero, making some features obsolete.
    This can be interpreted as a kind of feature selection.
    L2 Regularization, also called Ridge Regression (in context of regressions),
    shrinks the size of parameters, but not making them zero.
    In contrast to L1, L2 does not make the features obsolete, but reduces their impact.

    Parameters
    ----------
    input_size : int
        Input size for this layer
    output_size : int
        Output size equals the number of neurons per layer
    weight_regularizer_l1 : float, optional
        L1-Regularizer strength for the weights.
        If set to zero, no L1 regularization for the weights will be applied, by default 0
    weight_regularizer_l2 : float, optional
        L2-Regularizer strength for the weights.
        If set to zero, no L2 regularization for the weights will be applied, by default 0
    bias_regularizer_l1 : float, optional
        L1-Regularizer strength for the bias.
        If set to zero, no L1 regularization for the bias will be applied, by default 0
    bias_regularizer_l2 : float, optional
        L2-Regularizer strength for the bias.
        If set to zero, no L2 regularization for the bias will be applied, by default 0
    alpha : float, optional
        Decreases the value size of the weights during initialization to improve training, by default 0.01

    Attributes
    ----------
    weights : numpy.ndarray
        Weight matrix for the connections between input and output neurons.
    biases : numpy.ndarray
        Bias vector for the output neurons.
    inputs : numpy.ndarray
        The input values received during the forward pass.
    output : numpy.ndarray
        The output computed during the forward pass.
    dweights : numpy.ndarray
        Gradient of the loss with respect to weights.
    dbiases : numpy.ndarray
        Gradient of the loss with respect to biases.
    """

    def __init__(
            self,
            input_size:int,
            output_size:int,
            weight_regularizer_l1:float=0,
            weight_regularizer_l2:float=0,
            bias_regularizer_l1:float=0,
            bias_regularizer_l2:float=0,
            alpha:float=0.01,  
    ) -> None:
        
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Init weights
        self.weights = alpha * np.random.randn(input_size, output_size)

        # Init biases
        self.biases = np.zeros((1, output_size))
        # Alternatively:
        # self.biases = np.random.randn(output_size, 1)

        # Set regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    
    def forward(self, inputs:np.ndarray) -> None:
        """Computes a forward pass

        The formula can be vectorized to improve performance:
        :math:`output = inputs \cdot weights + biases`.
        
        Parameters
        ----------
        inputs : numpy.ndarray
            Input values from previous layer.
        """

        self.inputs = inputs

        # Perform a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues:np.ndarray) -> None:
        """Computes the backward step

        Parameters
        ----------
        dvalues : numpy.ndarray
            Derived gradient from the previous layer (reversed order).
        """

        # Compute gradient on weights & biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 

        # Include regularization for computed gradients

        # L1-Regularization
        # Weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # Biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2-Regularization
        # Weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # Biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Return parameters

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Returns on first index the weights and on second index the biases
        """
        return self.weights, self.biases

    def set_parameters(self, weights:np.ndarray, biases:np.ndarray) -> None:
        """Sets the parameters for this layer

        Parameters
        ----------
        weights : numpy.ndarray
            Weights of this layer.
        biases : numpy.ndarray
            Biases of this layer.
        """
        self.weights = weights
        self.biases = biases