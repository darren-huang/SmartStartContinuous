from unittest import TestCase
from smartstart.utilities.numerical import path_mean_and_std_per_state
import numpy as np

class TestNumerical(TestCase):

    def test_std_one_dim(self):
        assert np.equal([0],
                        path_mean_and_std_per_state([[1], [1], [1], [1]])[0])

    def test_std_two_dim(self):
        assert np.equal([0,0],
                        path_mean_and_std_per_state([[1, 1], [1, 1], [1, 1], [1, 1]])[0]).all()

    def test_std_two_dim_moving(self):
        assert np.equal([0,0],
                        path_mean_and_std_per_state([[1, 10], [0, 9], [1, 8], [0, 9]])[0]).all()

    def test_std_two_dim_nonzero(self):
        assert .50 == path_mean_and_std_per_state([[1], [2], [4]])[0][0]

    def test_means(self):
        for i in range(1,50):
            expected_avg = i + 50
            path = [[0]]
            for j in range(expected_avg - i, expected_avg + i + 1):
                path.append([path[-1][0] + j])
            assert path_mean_and_std_per_state(path)[1][0] == expected_avg

