"""Convolutional layer for 1D, 2D and 3D
"""

from pyml.neural_network.layer.transformation import _Transformation, _TrainableTransformation
import numpy as np
import math

class Convolutional(_Transformation, _TrainableTransformation):
    """Convolutional layer for a neural network.

    This class implements a convolutional layer with options for various parameters like kernel shape,
    padding, and stride.
    It is used for performing convolution or cross-correlation operations.

    Parameters
    ----------
    in_channels : int
        The number of input channels or depth.
    out_channels : int
        The number of output channels.
    kernel_shape : int or tuple[int,int]
        The shape of the convolutional kernel.
        If it's an integer, it's treated as a square kernel. 
        If it's a tuple, it should specify the height and width of the kernel.
    conv_operation : str, optional
        The type of convolution operation, either 'convolution' or 'cross-correlation'.
        By default 'cross-correlation'.
        Not implemented yet.
    padding : int or tuple[int,int] or str, optional
        Padding to apply to the input.
        It can be an integer, a tuple of two integers for height and width padding,
        or one of the strings 'valid', 'full', or 'same'.
        By default 0.
    stride : int or tuple[int,int], optional
        The stride to use during convolution.
        It can be an integer or a tuple of two integers specifying the vertical and horizontal strides.
        By default 1.
    weight_regularizer_l1 : float, optional
        L1 regularization strength for kernel weights, by default 0.
    weight_regularizer_l2 : float, optional
        L2 regularization strength for kernel weights, by default 0.
    bias_regularizer_l1 : float, optional
        L1 regularization strength for biases, by default 0.
    bias_regularizer_l2 : float, optional
        L2 regularization strength for biases, by default 0.

    Attributes
    ----------
    kernel_shape : tuple[int, int, int, int]
        The shape of the convolutional kernel in the format (output channels, input channels, height, width).
    flip_kernel : bool
        True if using cross-correlation, False if using convolution.
    """

    def __init__(
        self,
        in_channels:int, 
        out_channels:int, 
        kernel_shape: int|tuple[int,int],
        conv_operation:str='cross-correlation', 
        padding:int|tuple[int,int]|str=0, 
        stride:int|tuple[int,int]=1,
        weight_regularizer_l1:float=0,
        weight_regularizer_l2:float=0,
        bias_regularizer_l1:float=0,
        bias_regularizer_l2:float=0,
    ) -> None:
        
        super().__init__()

        # Set channels
        # TODO : Remove variables later and access them only via kernel shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Set kernel size
        # TODO : Remove kernel shape later and access its only trough kernel.shape
        if isinstance(kernel_shape, int):
            self.kernel_shape = (self.out_channels, self.in_channels, kernel_shape, kernel_shape)
        else:
            if len(kernel_shape) == 2:
                self.kernel_shape = (self.out_channels, self.in_channels, kernel_shape[0], kernel_shape[1])
        self.num_kernels = self.out_channels 

        # Set convolution operation type
        self.flip_kernel = False
        if conv_operation != 'convolution':
            self.flip_kernel = True

        # Set striding
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Set padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        elif padding == 'valid':
            self.padding = (0, 0)
        elif padding == 'full':
            self.padding = (self.kernel_height - 1, self.kernel_width - 1)
        elif padding == 'same':
            self.padding = ((self.kernel_height - 1) / 2, (self.kernel_width - 1) / 2)
        

        # Init weights (kernels) and biases
        sd = 1.0 / math.sqrt(self.kernel_shape[0])
        self.weights = np.random.uniform(-sd, sd, self.kernel_shape)
        self.biases = np.random.uniform(-sd, sd, (self.out_channels, 1, 1))
        

        # Set regularization
        # TODO : set regularization terms in parent class
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    @staticmethod
    def apply_striding(X:np.ndarray, stride:int|tuple[int, int]=1) -> np.ndarray:
        """Performs striding on a matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix of the dimension (Number of Batches, Channels/Depth, Height, Width)
        stride : int or tuple[int, int], optional
            Specifies the iteration step size of rows and columns to keep.
            If set to 1, all columns and rows will be kept.
            Stride can either be a single int or a tuple, where the first parameter sets
            the stride for rows (height), and the second parameter sets the stride for columns (width).
            By default 1.

        Returns
        -------
        numpy.ndarray
            Input matrix with applied stride.
        """
        if isinstance(stride, int):
            vertical_stride = stride
            horizontal_stride = stride
        else:
            vertical_stride, horizontal_stride = stride
        
        if vertical_stride == 1 and horizontal_stride == 1:
            return X
        
        return X[..., ::vertical_stride, ::horizontal_stride]
    

    def apply_padding(self, X:np.ndarray) -> np.ndarray:
        """Apply padding to a given input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix of the shape (Number of Batches, Channels/Depth, Height, Width).

        Returns
        -------
        numpy.ndarray
            Input matrix with applied padding.
        """
        X_pad = np.pad(
            X, 
            ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
            'constant', 
            constant_values=0
        )
        return X_pad

    def apply_kernel(self, X_slice:np.ndarray, kernel:np.ndarray, bias:np.ndarray) -> int:
        """Apply the convolution operation to a slice of the input using a given kernel and bias.

        Parameters
        ----------
        X_slice : numpy.ndarray
            A slice of the input matrix that matches the shape of the convolutional kernel.
        kernel : numpy.ndarray
            The convolutional kernel.
        bias : numpy.ndarray
            The bias associated with the kernel.

        Returns
        -------
        int
            The result of applying the convolution operation to the input slice using the given kernel and bias.
        """
        s = np.multiply(X_slice, kernel) + bias
        s = np.sum(s)

        return s

    def forward(self, inputs:np.ndarray) -> None:
        """Perform the forward pass through the convolutional layer.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data of the shape (Number of Batches, Channels/Depth, Height, Width).
        """

        self.inputs = inputs

        (in_num_batches, in_channels, in_height, in_width) = inputs.shape

        assert in_channels == self.in_channels, 'Number of channels of input does not match number of specified number of channels'

        vertical_padding, horizontal_padding = self.padding
        vertical_stride, horizontal_stride = self.stride
        _, _, kernel_height, kernel_width = self.kernel_shape

        out_height = int((in_height - self.kernel_shape[2] + 2 * vertical_padding) / vertical_stride) + 1
        out_width = int((in_width - self.kernel_shape[3] + 2 * horizontal_padding) / horizontal_stride) + 1

        output = np.zeros((in_num_batches, self.out_channels, out_height, out_width))

        X_pad = self.apply_padding(inputs)

        # Iterate through batches
        for i in range(in_num_batches):
            # Iterate through output channels
            for c in range(self.out_channels):
                # Iterate through output height
                for h in range(out_height):
                    # Iterate through output witdh
                    for w in range(out_width):
                        vertical_start = h * vertical_stride
                        vertical_end = vertical_start + kernel_height
                        horizontal_start = w * horizontal_stride
                        horizontal_end = horizontal_start + kernel_width
                        X_slice = X_pad[i, :, vertical_start:vertical_end, horizontal_start:horizontal_end]
                        output[i, c, h, w] = self.apply_kernel(X_slice, self.weights[c, ...], self.biases[c, ...])

        self.output = output


    def backward(self, dvalues:np.ndarray) -> None:
        """Perform the backward pass through the convolutional layer.

        Parameters
        ----------
        dvalues : numpy.ndarray
            The gradients of the loss with respect to the layer's output.
        """

        (in_num_batches, in_channels, in_height, in_width) = self.inputs.shape
        (out_num_batches, out_channels, out_height, out_width) = dvalues.shape

        assert in_num_batches == out_num_batches, 'Batch sizes of inputs and gradients are not equal'

        vertical_padding, horizontal_padding = self.padding
        vertical_stride, horizontal_stride = self.stride
        _, _, kernel_height, kernel_width = self.kernel_shape


        # Init gradients
        dweights = np.zeros(self.kernel_shape)
        dbiases = np.zeros(self.biases.shape)
        dinputs = np.zeros(self.inputs.shape)

        # Add padding
        X_pad = self.apply_padding(self.inputs)
        dinputs_pad = self.apply_padding(dinputs)

        # Iterate through batches
        for i in range(out_num_batches):
            i_X_pad = X_pad[i]
            i_dinputs_pad = dinputs_pad[i]
            # Iterate through gradient channels
            for c in range(out_channels):
                # Iterate through gradient height
                for h in range(out_height):
                    # Iterate through gradient witdh
                    for w in range(out_width):

                        vertical_start = h * vertical_stride
                        vertical_end = vertical_start + kernel_height
                        horizontal_start = w * horizontal_stride
                        horizontal_end = horizontal_start + kernel_width

                        x_slice = i_X_pad[:, vertical_start:vertical_end, horizontal_start:horizontal_end]

                        i_dinputs_pad[:, vertical_start:vertical_end, horizontal_start:horizontal_end] += self.weights[c, ...] * dvalues[i, c, h, w]
                        dweights[c, ...] += x_slice * dvalues[i, c, h, w]
                        dbiases[c, ...] += dvalues[i, c, h, w]

            # TODO : Optimize
            if vertical_padding == 0 and horizontal_padding == 0:
                dinputs[i, :, :, :] = i_dinputs_pad
            elif vertical_padding == 0 and horizontal_padding != 0:
                dinputs[i, :, :, :] = i_dinputs_pad[:, :, horizontal_padding:-horizontal_padding]
            elif vertical_padding != 0 and horizontal_padding == 0:
                dinputs[i, :, :, :] = i_dinputs_pad[:, vertical_padding:-vertical_padding, :]
            else:
                dinputs[i, :, :, :] = i_dinputs_pad[:, vertical_padding:-vertical_padding, horizontal_padding:-horizontal_padding]

        self.dweights = dweights
        self.dbiases = dbiases
        self.dinputs = dinputs

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Return parameters

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Returns on first index the weights (kernel) and on second index the biases
        """
        return self.weights, self.biases

    def set_parameters(self, weights:np.ndarray, biases:np.ndarray) -> None:
        """Sets the parameters for this layer

        Parameters
        ----------
        weights : numpy.ndarray
            Weights (kernel) of this layer.
        biases : numpy.ndarray
            Biases of this layer.
        """
        self.weights = weights
        self.biases = biases    