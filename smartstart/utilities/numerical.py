"""Module for numerical operations

"""
import numpy as np
import scipy.special

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

def dist_line_seg_to_point(line_seg_begin, line_seg_end, pt, dist_func, radii):
    # create vectors that have the line_seg_begin as the 'origin'
    pt_vector = pt - line_seg_begin
    line_vector = line_seg_end - line_seg_begin

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

def elliptical_euclidean_distance_scaled_function_generator(radii):
    """
      __
     / | \                    |
    |  | | where the vertical | represents the y-radius
    |  ._| and the horizontal _ represents the x-radius
    |    | (distance function makes all points on that ellipse become same distance from center)
     \__/  (the diagonal line segment 'radii' will be sacaled such that it is exactly 1 long)
    :param radii: for states of n-dimensions, radii is n - long specifying the elliptical radius along that axis
    :return:
    """
    for i in radii:
        assert i > 0
    radii = np.asarray(radii)
    scaling_term = 1
    def distance_func(state, other_state):
        """

        :param state: state/list of states (numpy) EITHER 1D or 2D
        :param other_state: state/list of states (numpy) EITHER 1D or 2D
        :return: distance/ list of distances (numpy)
        """
        axis = max(len(state.shape), len(other_state.shape)) - 1
        return np.sum((((state - other_state) / radii) / scaling_term) ** 2, axis=axis) ** 0.5

    scaling_term = distance_func(np.zeros(radii.shape), radii)

    return distance_func

def volume_of_n_dimensional_hyperellipsoid(radii):
    """
    n = dimension = len(radii)
    :param radii: the radii of each dimension (should be a list that is 'dim' long)
    :return: volume
    """
    dim = len(radii)
    return ((np.pi ** (dim / 2.)) / scipy.special.gamma((dim / 2.) + 1)) * np.product(radii)

def binary_search_index_lower(sorted_array, target, key = lambda x:x):
    """
    binary search, if equal, returns the index, otherwise return the index that is lower than given target,
    if nothing is lower, return None
    :param sorted_array:
    :param target:
    :param key:
    :return:
    """
    if key(sorted_array[0]) > key(target):
        return None
    start_i = 0
    end_i = len(sorted_array) - 1 # inclusive
    while start_i <= end_i:
        current_i = (start_i + end_i) // 2
        if key(sorted_array[current_i]) < key(target):
            start_i = current_i + 1
        elif key(sorted_array[current_i]) > key(target):
            end_i = current_i - 1
        else: #equal
            return current_i
    return end_i

def length_weighted_activities_solver(activities, sub_extra=0):
    """
    Solves the weighted activities problem where each activity's weight is end_time - start_time
    https://en.wikipedia.org/wiki/Activity_selection_problem
    :param activities: Should be an Nx2 array, should list of [x,y]s where 0 <= x < y
    :param sub_extra: (MUST BE POSITIVE) extra amount to subtract
    :return:
    """
    activities = sorted(activities, key=lambda x: x[1]) # sort by end times
    a0 = activities[0]
    optimal_tuple_list = [(0, 0, None, 0),(a0[1], a0[1] - a0[0], a0, 0)]
    # ordered by 1st index -> (end_time, optimal weight given the activities until end_time, activity to add, index of previous activities in optimal_tuple_list)
    for a in activities[1:]:
        latest_allowable_end_time_index = binary_search_index_lower(optimal_tuple_list, (a[0], None), key=lambda x: x[0]) #should never return None
        if a[1] == optimal_tuple_list[-1][0]: #end time same as previous end time
            optimal_tuple_list[-1] = max((a[1], optimal_tuple_list[latest_allowable_end_time_index][1] + (a[1] - a[0] - sub_extra), a, latest_allowable_end_time_index), #include activity 'a'
                                         (a[1], optimal_tuple_list[-1][1], optimal_tuple_list[-1][2], optimal_tuple_list[-1][3]),
                                         key=lambda x: x[1]) # previous calculation
        else:
            optimal_tuple_list.append(max((a[1], optimal_tuple_list[latest_allowable_end_time_index][1] + (a[1] - a[0] - sub_extra), a, latest_allowable_end_time_index), #include activity 'a'
                                         (a[1], optimal_tuple_list[-1][1], None, len(optimal_tuple_list) - 1),
                                          key=lambda x: x[1])) # previous calculation)  # previous calculation
    index = len(optimal_tuple_list) - 1
    selected_intervals = []
    if optimal_tuple_list[index][2] is not None:
        selected_intervals.append(optimal_tuple_list[index][2])
    while optimal_tuple_list[index][3] != index:
        index = optimal_tuple_list[index][3]
        if optimal_tuple_list[index][2] is not None:
            selected_intervals.append(optimal_tuple_list[index][2])
    selected_intervals.reverse()
    return optimal_tuple_list[-1][1], selected_intervals



def path_shortcutter(path, distance_func, theta):
    """
    given a path, points that are more than 1 state away are put adjacent in the path (middle points excluded) if they are theta away from each other
    ie. euclidean distance function, theta = 1, [[0],[2],[.5],[.6],[1.7],[2.9]] ->[[0],[.6],[1.7],[2.9]]
    :param path: list of states (this is the sequence of states that make up the path)
    :param distance_func: takes in two states, returns a scalar, if given a list of states (2d numpy array) then
            will compute a list of distances, if given two lists of states (1xnxd) and (nx1xd) where n is the number of states and d is the dimension of the states,
            will return a nxnx1 matrix of distances between each states pairwise
    :param theta: minimum distance between two states to shortcut ie. theta = 1, [1, 3, 1.5, 1.2, 2.1, 3.2] shortcuts to [1, 1.2, 2.1, 3.2]
    :return: new path
    """
    a = np.array(path)
    ar = a.reshape((1, *a.shape))
    distances_arr = distance_func(ar.transpose((1,0,2)), ar)
    indices = np.transpose(np.where(np.triu(distances_arr <= theta, k=2))) # can't shortcut (1,2) since there is no index 1.5 in between, we are using k=2 instead of k=1
    length_of_intervals, selected_intervals = length_weighted_activities_solver(indices.tolist(), sub_extra=1) #sub_extra=1 because intervale (1,4) only removes index 2 and 3 therefore weight should be '2' not '4-1'
    indices_to_delete = []
    for indices_pair in selected_intervals:
        indices_to_delete.extend(range(indices_pair[0] + 1, indices_pair[1])) # want to include start and end
    new_path = np.delete(a, indices_to_delete, axis=0)
    return new_path

