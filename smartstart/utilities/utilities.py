"""Module that contains random utility functions used in the smartstart package

"""
import os
import random
import glob

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                   'data/default')
if not os.path.exists(DIR):
    os.makedirs(DIR)

def get_glob_summaries(file_name):
    from smartstart.utilities.datacontainers import Summary

    fps = glob.glob(file_name)
    summaries = [Summary.load(fp) for fp in fps]
    return summaries


def get_default_data_directory(foldername, create=True):
    """
    Used for getting a filepath to a default data directory which is right next to the smartstart source folder
    :param foldername: name of the data folder
    :return: directory path
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                 'data/')
    path = os.path.join(path, foldername)
    if (not os.path.exists(path)) and create:
        os.makedirs(path)
    return path

def get_default_model_directory(foldername, create=True):
    """
    Used for getting a filepath to a default model directory which is right next to the smartstart source folder
    :param foldername: name of the model folder
    :return: directory path
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../',
                        'models/')
    path = os.path.join(path, foldername)
    if (not os.path.exists(path)) and create:
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
    """
    Get a list of waypoints including the starting state and final state, where between each waypoint is
    'steps_per_waypoints' steps apart
    :param path:
    :param steps_per_waypoint:
    :return:
    """
    if isinstance(path, np.ndarray):
        path = path.tolist()
    waypoint_centers = path[:-1:steps_per_waypoint]
    waypoint_centers.append(path[-1])
    return waypoint_centers

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)