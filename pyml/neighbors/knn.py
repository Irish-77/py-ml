"""k-nearest neighbors (kNN) classifier"""

import numpy as np

from pyml.exceptions import ShapeError
from pyml.utils.metrics import manhatten_distance, euclidean_distance


class UnknownMetric(Exception):
    """Exception raised for unknown metric methods
    
    Parameters
    ---------
    metric : str
        Name of the metric provided by the user

    Examples
    --------
    >>> from pyml.neighbors import kNNClassifier
    >>> model = kNNClassifier(metric='abc metric')
    The metric "abc metric" is not implemented. Please refer to the documentation.
    """

    def __init__(self, metric:str) -> None:
        self.message = f'The metric "{metric}" is not implemented. Please refer to the documentation.'
        super().__init__(self.message)

class kNNClassifier():
    """Classifier model using the nearest neighbor algorithm

    K-nearest neighbor (KNN) is a simple and intuitive machine learning algorithm,
    that can be used for classification and regression tasks.
    In the case of classification the model predicts the class of an data point based
    on the majority class or average of its K nearest data points in the feature space.

    Following metrics are support:
    - euclidean
    - manhatten

    Parameters
    ----------
    k : int, optional
        Specifies the number of nearest neighbor to consider when predicting on new data.
        By default 3.
    metric : str, optional
        Specifies the metric used for calculating the distance
        By default 'euclidean'.

    Attributes
    ----------
    metrics : List[str]
        Defines the metrics that are currently supported
    
    Raises
    ------
    UnknownMetric
        Raised when using an unknow metric name (including spelling errors)
    ShapeError
        Raised when computing the distance for incompatible matrices
    """

    metrics = ['euclidean', 'manhatten']

    def __init__(self, k:int=3, metric:str='euclidean') -> None:
        
        self.k = k

        if metric not in self.metrics:
            raise UnknownMetric(metric)
        else:
            self.metric = metric

    def _compute_distance(
            self,
            x1: np.ndarray,
            x2: np.ndarray
    ) -> np.array:
        """Computes the distance between two matrix-like objects using the defined metric

        One of the parameters must be a matrix with only one row or alternativly a vector.

        Parameters
        ----------
        x1 : numpy.ndarray
            Input matrix
        x2 : numpy.ndarray
            Input matrix

        Returns
        -------
        numpy.ndarray
            Matrix consisting of the distances

        Raises
        ------
        ShapeError
            If shapes do not match a shape error

        See Also
        --------
        pyml.exceptions.ShapeError
        """
        
        if self.metric == 'euclidean':
            dist_func = euclidean_distance
        elif self.metric == 'manhatten':
            dist_func = manhatten_distance
        
        try:
            distance = dist_func(x1, x2)
        except:
            raise ShapeError(x1.shape, x2.shape)
            
        return distance
    
    def fit(
            self,
            X:np.ndarray,
            y:np.array
    ) -> None:
        """Fit model on training data

        Since the k nearest neighbor algorithm is a lazy learner, there will be no training.
        However, the training data will be stored in memory.

        Parameters
        ----------
        X : numpy.ndarray
            Input training data
        y : numpy.array
            Input training labels
        """
        
        self.X = np.atleast_2d(X)
        self.y = y

    def predict(
            self,
            X:np.ndarray,
            return_class_prob:bool=False
    ) -> np.array:
        """Calculates predictions for given data points

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix; for each row the k nearest neighbor is being calculated
        return_class_prob : bool, optional
            If set to true, the respective probability of each prediction is be returned as well
            (#predicted_class / k).
            By default False.

        Returns
        -------
        numpy.ndarray
            Returns predicted labels and if specified their respective probability.
        """

        # Add one dimension in case input matrix is of shape (n, )
        X = np.atleast_2d(X)
        
        # Compute distances for each input entry point
        distances = np.apply_along_axis(func1d = self._compute_distance, axis=1, arr=X, x2=self.X)

        # Get indices of datapoints with shortest distance
        indices = np.apply_along_axis(func1d = np.argsort, axis=1, arr=distances)

        # Get the respective classes
        nearest_k_classes = self.y[indices[:, :self.k]]

        # Retrieve the frequency of the nearest classes
        axis = 1
        classes, indices = np.unique(nearest_k_classes, return_inverse=True)
        class_predictions = classes[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(nearest_k_classes.shape), None, np.max(indices) + 1), axis=axis)]

        if return_class_prob:
            frequencies = np.empty(X.shape[0], dtype=float)
            for i, row in enumerate(X):
                _, counts = np.unique(row, return_counts=True)
                frequencies[i] = counts.max() / self.k

            return class_predictions, frequencies
        
        return class_predictions