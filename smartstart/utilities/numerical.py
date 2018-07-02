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
def path_mean_and_std_per_state(path):
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

