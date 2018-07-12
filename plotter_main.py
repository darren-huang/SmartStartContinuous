from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.plot import plot_path, plot_summary, \
    mean_reward_episode, steps_episode, show_plot, total_rewards_episode, \
    plot_get_axes
from smartstart.utilities.utilities import get_default_directory, get_start_waypoints_final_states_steps
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc

import os


target_default_directory = "ddpg_summaries"
# target_file_name = "DDPG_agent_MountainCarContinuous-v0-1000ep.json"
target_file_name = "DDPG_agent_MountainCarContinuous-v0_test-2ep.json"

if __name__ == "__main__":
    target_path = os.path.join(get_default_directory(target_default_directory), target_file_name)
    summary = Summary.load(target_path)

    print("Saved the last " + str(summary.last_x) + " episodes")


    # Settings##############################
    # waypoint size
    linewidth = 3
    num_stds = 1
    num_steps = 1
    num_means = 1
    steps_per_waypoint = 1



    ########################################

    # Plot results
    ((ax1, ax2), (ax3, ax4)) = plot_get_axes(2,2)
    plot_summary(summary, mean_reward_episode, ma_window=1, axis=ax1,
                 title="MountainCarContinuous-v0 Q-Learning Average Reward per Episode")
    plot_summary(summary, steps_episode, ma_window=1, axis=ax2,
                 title="MountainCarContinuous-v0 Q-Learning Steps per Episode")
    plot_summary(summary, total_rewards_episode, ma_window=1, axis=ax3,
                 title="MountainCarContinuous-v0 Q-Learning Total Reward per Episode")

    #LAST Trajectory##   ########################################################
    # which "last" trajectory to show
    last_index = 0
    path, reward, title = summary.get_last_path(last_index), summary.get_last_reward(last_index), str(last_index) + " From Last Path"
    waypoint_centers = get_start_waypoints_final_states_steps(path, steps_per_waypoint)
    path_stds, path_means = path_deltas_stds_and_means_per_dim(path)  # Get State Statistics
    print("Last Traj: STD's: {}, Means: {}".format(path_stds, path_means))
    radii = (path_means + (path_stds * num_stds)) * num_steps  # radii for waypoints
    ## Plot Paths/Trajectories
    plot_path(path,
              title=title,
              reward=reward,
              x_label="x_pos",
              y_label="x_velocity",
              waypoint_centers=waypoint_centers,
              radii=radii,
              linewidth=linewidth)


    #BEST Trajectory ##########################################################
    # number of waypoints for BEST PATH
    path, reward, title = summary.best_path, summary.best_reward, "Best Path"
    waypoint_centers = get_start_waypoints_final_states_steps(path, steps_per_waypoint)
    path_stds, path_means = path_deltas_stds_and_means_per_dim(path) # Get State Statistics
    print("Best Traj: STD's: {}, Means: {}".format(path_stds, path_means))
    radii = radii_calc(path_means, path_stds, num_means, num_stds, num_steps) # radii for waypoints


    ## Plot Paths/Trajectories
    plot_path(path,
              title=title,
              reward=reward,
              x_label="x_pos",
              y_label="x_velocity",
              waypoint_centers=waypoint_centers,
              highlight_waypoint_index=0,
              radii=radii,
              linewidth=linewidth)


    show_plot()