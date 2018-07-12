from unittest import TestCase
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, \
    projection_of_a_onto_b, dist_line_seg_to_point
import numpy as np

class TestNumerical(TestCase):

    def test_std_one_dim(self):
        assert np.equal([0],
                        path_deltas_stds_and_means_per_dim([[1], [1], [1], [1]])[0])

    def test_std_two_dim(self):
        assert np.equal([0,0],
                        path_deltas_stds_and_means_per_dim([[1, 1], [1, 1], [1, 1], [1, 1]])[0]).all()

    def test_std_two_dim_moving(self):
        assert np.equal([0,0],
                        path_deltas_stds_and_means_per_dim([[1, 10], [0, 9], [1, 8], [0, 9]])[0]).all()

    def test_std_two_dim_nonzero(self):
        assert .50 == path_deltas_stds_and_means_per_dim([[1], [2], [4]])[0][0]

    def test_means(self):
        for i in range(1,50):
            expected_avg = i + 50
            path = [[0]]
            for j in range(expected_avg - i, expected_avg + i + 1):
                path.append([path[-1][0] + j])
            assert path_deltas_stds_and_means_per_dim(path)[1][0] == expected_avg

    def test_projection_of_a_onto_b_2d_simple(self):
        a = np.array([1,1])
        b = np.array([0,1])
        assert np.equal(projection_of_a_onto_b(a,b), np.array([0,1])).all()

    def test_projection_of_a_onto_b_3d_simple(self):
        a = np.array([1,1,1])
        b = np.array([0,1,1])
        assert np.allclose(projection_of_a_onto_b(a,b), np.array([0,1,1]))

    def test_dist_line_seg_to_point_simple(self):
        line_seg_begin = np.array([0,1])
        line_seg_end = np.array([1, 2])
        pt = np.array([2,1])
        assert np.isclose(dist_line_seg_to_point(line_seg_begin, line_seg_end, pt, [1,1]), 2 ** .5)

    def test_dist_line_seg_to_point_stretched(self):
        stretch = 10
        line_seg_begin = np.array([0,1*stretch])
        line_seg_end = np.array([1, 2*stretch])
        pt = np.array([2,1*stretch])
        assert np.isclose(dist_line_seg_to_point(line_seg_begin, line_seg_end, pt, [1,stretch]), 2 ** .5)

