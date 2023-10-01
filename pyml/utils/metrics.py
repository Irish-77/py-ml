"""Collection of common metrics used in machine learning"""

import numpy as np

def euclidean_distance(
            x1:np.ndarray,
            x2:np.ndarray
    ) -> np.ndarray:
        """Computes the euclidean distance for two matrix-like objects

        .. math::
            d(p,q)=\\|q-p\\|_{2}={\\sqrt {(q_{1}-p_{1})^{2}+\\cdots +(q_{n}-p_{n})^{2}}}={\\sqrt {\\sum _{i=1}^{n}(q_{i}-p_{i})^{2}}}
            
        Parameters
        ----------
        x1 : numpy.ndarray
            Input matrix
        x2 : numpy.ndarray
            Input matrix
            The shapes of x1 and x2 must be compatible in terms of matching column number.

        Returns
        -------
        numpy.ndarray
            computed distance using the euclidean metric
        """
    
        distances = np.sqrt(np.sum(np.square(np.subtract(x1, x2)), axis = 1))
    
        return distances

def manhatten_distance(
        x1:np.ndarray,
        x2:np.ndarray
) -> np.ndarray:
    
    """Computes the manhatten distance for two matrix-like objects

    .. math::
        d(A,B)=\\sum _{i}\\left|A_{i}-B_{i}\\right|

    Parameters
    ----------
    x1 : numpy.ndarray
        Input matrix
    x2 : numpy.ndarray
        Input matrix
        The shapes of x1 and x2 must be compatible in terms of matching column number.

    Returns
    -------
    numpy.ndarray
        computed distance using the manhatten metric
    """
    distances = np.sum(np.abs(np.subtract(x1, x2)), axis = 1)

    return distances
