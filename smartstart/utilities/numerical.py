"""Module for numerical operations

"""
import numpy as np


# Moving average
def moving_average(values, window=10):
    """Moving average filter

    Parameters
    ----------
    values : :obj:`np.ndarray`
        array the moving average filter has to applied on
    window : :obj:`int`
        window size for the moving average filter (Default value = 10)

    Returns
    -------
    :obj:`np.ndarray`
        values after applying moving average filter (length = orignal_length
        - window + 1)
    """
    if window == 1:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

# path state changes standard deviation
def path_deltas_stds_and_means_per_dim(path):
    """
    For each individual state value, calculates the standard deviation of delta's for that state (delta is the difference
    from one state value to the next)
    NOTE: delta's are always positive

    :param path: :obj: array
        array of states that make up a path
            a state  should be a 1 dimensional array of state values
    :return:
        a std deviation of the change in each state per step, ie. for the "path" [[0],[1],[2],[3],[4]]
            The changes per step would be [1,1,1,1] since there are 4 steps, where statevalue0 changes by 1 each time
            Std deviation would be 0
            Since the state has only 1 variable, the return is [0.0]
    """
    if len(path) <= 1:
        return "how about no"

    state_dim = len(path[0])
    individual_state_value_deltas = [[] for _ in range(state_dim)]
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i+1]
        for j in range(state_dim):
            individual_state_value_deltas[j].append(abs(next_state[j] - current_state[j]))

    stds = np.asarray([np.std(delta_arr) for delta_arr in individual_state_value_deltas])
    means = np.asarray([np.mean(delta_arr) for delta_arr in individual_state_value_deltas])

    return stds, means

def radii_calc(means, stds, num_means, num_stds, num_steps):
    return ((num_means * means) + (num_stds * stds)) * num_steps

def euclidean_distance(state, other_state):
    """

    :param state: state/list of states (numpy) EITHER 1D or 2D
    :param other_state: state/list of states (numpy) EITHER 1D or 2D
    :return: distance/ list of distances (numpy)
    """
    axis = max(len(state.shape), len(other_state.shape)) - 1
    return np.sum((state - other_state) ** 2, axis=axis) ** 0.5

def dist_line_seg_to_point(line_seg_begin, line_seg_end, pt, radii):
    # create vectors that have the line_seg_begin as the 'origin'
    pt_vector = pt - line_seg_begin
    line_vector = line_seg_end - line_seg_begin
    dist_func = elliptical_euclidean_distance_function_generator(radii)

    # note the returned projection is a also a vector with the line_seg_begin as the 'origin'
    return dist_func(projection_of_a_onto_b(pt_vector, line_vector, radii = radii),
                     pt_vector)


def projection_of_a_onto_b(a, b, radii=None):
    if radii is not None:
        # scale a and b
        a, b = a / radii, b/ radii

    a_dot_b = np.sum(a*b)
    b_magnitude = (np.sum(b * b) ** 0.5)
    c = a_dot_b / b_magnitude # c*b_hat = projection component where b_hat is the unit vector of b
    b_hat = b / (np.sum(b * b) ** 0.5)
    proj_of_a_onto_b = c * b_hat

    if radii is not None:
        # unscale projection back to normal basis
        proj_of_a_onto_b = proj_of_a_onto_b * radii
    return proj_of_a_onto_b


def elliptical_euclidean_distance_function_generator(radii):
    """
      __
     / | \                    |
    |  | | where the vertical | represents the y-radius
    |  ._| and the horizontal _ represents the x-radius
    |    | (distance function makes all points on that ellipse become distance 1 from the center
     \__/
    :param radii: for states of n-dimensions, radii is n - long specifying the elliptical radius along that axis
    :return:
    """
    for i in radii:
        assert i > 0
    radii = np.asarray(radii)

    def distance_func(state, other_state):
        """

        :param state: state/list of states (numpy) EITHER 1D or 2D
        :param other_state: state/list of states (numpy) EITHER 1D or 2D
        :return: distance/ list of distances (numpy)
        """
        axis = max(len(state.shape), len(other_state.shape)) - 1
        return np.sum(((state - other_state) / radii) ** 2, axis=axis) ** 0.5

    return distance_func