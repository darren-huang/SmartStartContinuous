from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.plot import plot_path, plot_summary, \
    mean_reward_episode, steps_episode, show_plot, total_rewards_episode, \
    plot_get_axes
from smartstart.utilities.utilities import get_default_directory
from smartstart.utilities.numerical import path_mean_and_std_per_state

import os


target_default_directory = "ddpg_summaries"
target_file_name = "DDPG_agent_MountainCarContinuous-v0-1000ep.json"

if __name__ == "__main__":
    target_path = os.path.join(get_default_directory(target_default_directory), target_file_name)
    summary = Summary.load(target_path)

    print("Saved the last " + str(summary.last_x) + " episodes")
    def last_path(x): # x goes from 0 to last_x - 1
        return summary.last_paths[-(x + 1)]
    def last_reward(x):# x goes from 0 to last_x - 1
        return summary.last_rewards[-(x + 1)]

    # Settings##############################
    # waypoint size
    linewidth = 2
    num_stds = 1
    num_steps = 1



    ########################################

    # Plot results
    ((ax1, ax2), (ax3, ax4)) = plot_get_axes(2,2)
    plot_summary(summary, mean_reward_episode, ma_window=1, axis=ax1,
                 title="MountainCarContinuous-v0 Q-Learning Average Reward per Episode")
    plot_summary(summary, steps_episode, ma_window=1, axis=ax2,
                 title="MountainCarContinuous-v0 Q-Learning Steps per Episode")
    plot_summary(summary, total_rewards_episode, ma_window=1, axis=ax3,
                 title="MountainCarContinuous-v0 Q-Learning Total Reward per Episode")

    #LAST Trajectory
    # which "last" trajectory to show
    last_index = 0
    # number of waypoints for BEST PATH
    steps_per_waypoint = 20
    num_waypoints = len(last_path(last_index)) // steps_per_waypoint
    # Get State Statistics
    path_stds, path_means = path_mean_and_std_per_state(last_path(last_index))
    print("Last Traj: STD's: {}, Means: {}".format(path_stds, path_means))

    # Plot Paths/Trajectories
    radii = (path_means + (path_stds * num_stds)) * num_steps

    #last trajectory
    plot_path(last_path(last_index),
              title=str(last_index) + " From Last Path",
              reward=last_reward(last_index),
              x_label="x_pos",
              y_label="x_velocity",
              num_waypoints=num_waypoints,
              radii=radii,
              linewidth=linewidth)

    #BEST Trajectory
    # number of waypoints for BEST PATH
    steps_per_waypoint = 10
    num_waypoints = len(summary.best_path) // steps_per_waypoint
    # Get State Statistics
    path_stds, path_means = path_mean_and_std_per_state(summary.best_path)
    print("Best Traj: STD's: {}, Means: {}".format(path_stds, path_means))

    # Plot Paths/Trajectories
    radii = (path_means + (path_stds * num_stds)) * num_steps
    #best trajectory
    plot_path(summary.best_path,
              title="Best Path",
              reward=summary.best_reward,
              x_label="x_pos",
              y_label="x_velocity",
              num_waypoints=num_waypoints,
              radii=radii,
              linewidth=linewidth)


    show_plot()