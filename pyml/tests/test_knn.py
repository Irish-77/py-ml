"""
Run this test by exceuting following command
```
python -m pyml.tests.test_knn
```
"""

import unittest
import numpy as np

from pyml.neighbors import kNNClassifier, UnknownMetric
from pyml.exceptions import ShapeError

class TestkNNClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = kNNClassifier(k=5)

        # Create data
        cluster1 = [[1, 1, 3, 'A'], [1, 2, 1, 'A'], [2, 1, 4, 'A'], [2, 2, -1, 'A']]
        cluster2 = [[-1, 1, 3, 'B'], [-1, 2, 1, 'B'], [-2, 1, 4, 'B'], [-2, 2, -1, 'B']]
        cluster3 = [[-1, -1, 3, 'C'], [-1, -2, 1, 'C'], [-2, -1, 4, 'C'], [-2, -2, -1, 'C']]
        cluster4 = [[1, -1, 3, 'D'], [1, -2, 1, 'D'], [2, -1, 4, 'D'], [2, -2, -1, 'D']]

        self.data = np.vstack([cluster1, cluster2, cluster3, cluster4])
        self.X = self.data[:, :-1].astype(np.float32)
        self.y = self.data[:, -1]

        self.model.fit(self.X, self.y)

    def test_single_prediction(self):
        new_data = np.array([-0.5, 2, 3])
        actual_prediction = ['B']

        prediction = self.model.predict(new_data)

        self.assertEqual(actual_prediction, prediction)

    def test_single_prediction_with_class_probability(self):
        new_data = np.array([1, 2, 4])
        actual_prediction = (['A'], [0.2])

        prediction = self.model.predict(new_data, True)

        self.assertEqual(actual_prediction, prediction)

    def test_multiple_predictions(self):
        new_data = np.array(
            [
                [-1, -1, 3],
                [1, -2, 3]
            ]
        )
        actual_prediction = ['C', 'D']

        prediction = self.model.predict(new_data)

        for a, b in zip(actual_prediction, prediction):
            self.assertEqual(a, b, 1)

class TestkNNClassifierMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.model = kNNClassifier(k=5)

    def test_manhatten_distance(self):
        x1 = np.array([[1, 1, 1]])
        x2 = np.array([[0, 0, 0], [1, 1, 1], [1, 2, 3]])

        actual_distances = np.array([3, 0])

        self.model.metric = 'manhatten'

        result_distances = self.model._compute_distance(x1, x2)
        for a, b in zip(actual_distances, result_distances):
            self.assertAlmostEqual(a, b, 1)
    
    def test_euclidean_distance(self):

        self.model.metric = 'euclidean'

        x1 = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1, -0.1],
                [1200, 1300, -943.1, 38.87],
                [0, 0, 0, 0],
            ]
        )

        x2 = np.array([1, 2, 3, 4])
        x3 = np.array(
            [
                [1, 2, 3, 4],
                [0, 0, 2, 0]
            ]
        )

        actual_distance_x2 = np.array([5.3, 5.52, 2004.68, 5.48])

        # x1, x2
        result_distance = self.model._compute_distance(x1, x2)
        for a, b in zip(actual_distance_x2, result_distance):
            self.assertAlmostEqual(a, b, 1)

        # x1, [x2]
        result_distance = self.model._compute_distance(x1, [x2])
        for a, b in zip(actual_distance_x2, result_distance):
            self.assertAlmostEqual(a, b, 1)

        # x2, x1
        result_distance = self.model._compute_distance(x2, x1)
        for a, b in zip(actual_distance_x2, result_distance):
            self.assertAlmostEqual(a, b, 1)

    def test_shape_error(self):
        x1 = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1, -0.1],
                [1200, 1300, -943.1, 38.87],
                [0, 0, 0, 0],
            ]
        )
        x3 = np.array(
            [
                [1, 2, 3, 4],
                [0, 0, 2, 0]
            ]
        )
        actual_distance_x3 = np.array(
            [
                [5.3, 5.52, 2004.68, 5.48],
                [1.91, 1.91, 2006.17, 2]
            ]
        )
        with self.assertRaises(ShapeError):
            result_distance = self.model._compute_distance(x1, x3)
            for a, b in zip(actual_distance_x3, result_distance):
                self.assertAlmostEqual(a, b, 1)

    def test_unknown_distance(self):
        with self.assertRaises(UnknownMetric):
            kNNClassifier(k=1, metric='invalid distance')


if __name__ == '__main__':
    unittest.main(verbosity=2)

# https://docs.python.org/3/library/unittest.html