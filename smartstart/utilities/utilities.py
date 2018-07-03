"""Module that contains random utility functions used in the smartstart package

"""
import os
import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                   'data/default')
if not os.path.exists(DIR):
    os.makedirs(DIR)


def get_data_directory(file):
    """Creates and returns a data directory at the file's location

    Parameters
    ----------
    file :
        python file

    Returns
    -------
    :obj:`str`
        filepath to the data directory

    """
    fp = os.path.dirname(os.path.abspath(file))
    fp = os.path.join(fp, 'data')
    if not os.path.exists(fp):
        os.makedirs(fp)
    return fp

def get_default_directory(foldername):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                 'data/')
    path = os.path.join(path, foldername)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_start_waypoints_final_states(path, num_waypoints):
    """
    Given a path, returns a list consisting of the first state, 'num_waypoints' of waypoints, and the final state
    :param path: the paty to select waypoints from
    :param num_waypoints: waypoints on the path
    :return: list of states (includes start state and end state
    """
    return get_start_waypoints_final_states_steps(path, (len(path) - 1) // num_waypoints)


def get_start_waypoints_final_states_steps(path, steps_per_waypoint):
    waypoint_centers = path[:-1:steps_per_waypoint]
    waypoint_centers.append(path[-1])
    return waypoint_centers