import random

import numpy as np


from smartstart.RLDiscreteAlgorithms.qlearning import QLearning
from smartstart.environments.gridworld import GridWorld
from smartstart.environments.gridworldvisualizer import GridWorldVisualizer
from smartstart.utilities.plot import plot_summary, show_plot, \
    mean_reward_episode, steps_episode
from reinforcementLearningCore.rlTrain import rlTrain

# Reset the seed for random number generation
random.seed()
np.random.seed()

# Create environment and visualizer
grid_world = GridWorld.generate(GridWorld.EASY)
visualizer = GridWorldVisualizer(grid_world)
visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                          GridWorldVisualizer.CONSOLE,
                          GridWorldVisualizer.VALUE_FUNCTION,
                          GridWorldVisualizer.DENSITY)

# Initialize agent, see class for available parameters
agent = QLearning(grid_world,
                  alpha=0.1,
                  epsilon=0.05,
                  exploration=QLearning.E_GREEDY)

# Train the agent, summary contains training data
summary = rlTrain(agent, grid_world, render=True,
                  render_episode=False,
                  print_results=True, num_episodes=500, max_steps=1000)

# Plot results
plot_summary(summary, mean_reward_episode, ma_window=5, title="Easy GridWorld Q-Learning Average Reward per Episode")
plot_summary(summary, steps_episode, ma_window=5, title="Easy GridWorld Q-Learning Steps per Episode")
show_plot()
