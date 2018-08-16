import random

import numpy as np

from smartstart.RLDiscreteAlgorithms.qlearning import QLearning
# from smartstart.smartexploration.smartexplorationdiscrete import generate_smartstart_object
from smartstart.environments.gridworld import GridWorld
from smartstart.utilities.experimenter import run_experiment
from smartstart.utilities.utilities import get_default_data_directory

# Get the path to the data folder in the same directory as this file.
# If the folder does not exists it will be created
summary_dir = get_default_data_directory("")


# Define the task function for the experiment
def task(params):
    print(params)


# Define a parameter grid that can be supplied to the run_experiment method
param_grid = {
    'task': task,
    'num_exp': 5,
    'use_smart_start': [True, False],
    'asdf' : [0, 1]
}

if __name__ == '__main__':
    run_experiment(param_grid, n_processes=-1)
