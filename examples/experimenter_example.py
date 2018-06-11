import random

import numpy as np

from smartstart.RLDiscreteAlgorithms.qlearning import QLearning
from smartstart.smartexploration.smartexplorationdiscrete import generate_smartstart_object
from smartstart.environments.gridworld import GridWorld
from smartstart.utilities.experimenter import run_experiment
from smartstart.utilities.utilities import get_data_directory

# Get the path to the data folder in the same directory as this file.
# If the folder does not exists it will be created
summary_dir = get_data_directory(__file__)


# Define the task function for the experiment
def task(params):
    # Reset the seed for random number generation
    random.seed()
    np.random.seed()

    # Create environment
    env = GridWorld.generate(GridWorld.MEDIUM)

    # Here we use a dict to define the parameters, this makes it easy to
    # make sure the experiments use the same parameters
    kwargs = {
        'alpha': 0.1,
        'epsilon': 0.05,
        'num_episodes': 1000,
        'max_steps': 2500,
        'exploration': QLearning.E_GREEDY
    }

    # Initialize agent, check params if it needs to use SmartStart or not
    if params['use_smart_start']:
        agent = generate_smartstart_object(QLearning, env, **kwargs)
    else:
        agent = QLearning(env, **kwargs)

    # Train the agent, summary contains training data. Make sure to set the
    # rendering and printing to False when multiple experiments run in
    # parallel. Else it will consume a lot of computation power.
    summary = agent.train(render=False,
                          render_episode=False,
                          print_results=False)

    # Save the summary. The post_fix parameter can be used to create a unique
    #  file name.
    summary.save(directory=summary_dir, post_fix=params['run'])


# Define a parameter grid that can be supplied to the run_experiment method
param_grid = {
    'task': task,
    'num_exp': 5,
    'use_smart_start': [True, False]
}

if __name__ == '__main__':
    run_experiment(param_grid, n_processes=-1)
