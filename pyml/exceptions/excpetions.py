"""Python module that contains exceptions/errors used across several classes
"""

class ShapeError(Exception):
    """Exception raised when performing a matrix/vector operation on two objects with non-compatible sizes
    
    Examples
    --------
    >>> from pyml.neighbors import kNNClassifier
    >>> model = kNNClassifier()
    >>> x = np.random.randn(2, 3)
    >>> y = np.random.randn(3, 4)
    >>> model._compute_distance(x, y)
    The matrix operations for objects of the size (2,3), (3,2) can not be computed.
    """

    def __init__(self, *args: tuple) -> None:
        self.message = f'The matrix operations for objects of the size "{args}" can not be computed.'
        super().__init__(self.message)
        

class OutsideSpecifiedRange(Exception):
    """Exception raised when an input value is outside the specified range.

    This exception is raised when a provided input value is not within the limits
    defined by a specified lower and upper bound.

    Parameters
    ----------
    input : float
        The input value that is outside the specified range.
    variable_name : str
        The name of the variable associated with the input value.
    lower_limit : float
        The lower limit of the specified range.
    upper_limit : float
        The upper limit of the specified range.
    """

    def __init__(self, input:float, variable_name:str, lower_limit:float, upper_limit:float) -> None:
        self.message = f'Your input {input} for the variable {variable_name} is not within the specified limits {lower_limit} and {upper_limit}.'
        super().__init__(self.message)

class HyperparametersNotSpecified(Exception):
    """Excpetion raised when a model is initialized or training is started but hyperparameters are not specified yet.

    Parameters
    ----------
    hyperparameter : str
        Name of the hyperparameter.
    """

    def __init__(self, hyperparameter:str):
        self.message = f'You havn\'t specified the hyperparameter: {hyperparameter}.'
        super().__init__(self.message)