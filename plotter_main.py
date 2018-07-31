from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.plot import plot_path, plot_summary, \
    mean_reward_episode, steps_episode, show_plot, total_rewards_episode, \
    plot_get_axes_and_fig
from smartstart.utilities.utilities import get_default_data_directory, get_start_waypoints_final_states_steps
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc

import os

def plot_file(target_default_directory, target_file_name, first_num_episodes=None, paths = True):
    target_path = os.path.join(get_default_data_directory(target_default_directory), target_file_name)
    summary = Summary.load(target_path)

    if first_num_episodes:
        summary.episodes = summary.episodes[:first_num_episodes]

    print("Saved the last " + str(summary.last_x) + " episodes")


    # Settings##############################
    # waypoint size
    linewidth = 3
    num_stds_per_step = 1
    num_means_per_step = 1
    num_steps_in_waypoint_radii = 1
    steps_between_waypoints = 5



    ########################################
    name = target_file_name.replace(".json","")
    # Plot results
    ((ax1, ax2), (ax3, ax4)), fig = plot_get_axes_and_fig(2,2)
    plot_summary(summary, mean_reward_episode, ma_window=1, axis=ax1, first_num_episodes = first_num_episodes,
                 title="Average Reward per Episode\n" + name, dots = True)
    plot_summary(summary, steps_episode, ma_window=1, axis=ax2, first_num_episodes = first_num_episodes,
                 title="Steps per Episode\n" + name, dots = True)
    plot_summary(summary, total_rewards_episode, ma_window=1, axis=ax3, first_num_episodes = first_num_episodes,
                 title="Total Reward per Episode\n" + name, dots = True)
    fig.tight_layout()

    if paths:
        #LAST Trajectory##   ########################################################
        # which "last" trajectory to show
        last_index = 0
        path, reward, title = summary.get_last_path(last_index), summary.get_last_reward(last_index), str(last_index) + " From Last Path"
        waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
        path_stds, path_means = path_deltas_stds_and_means_per_dim(path)  # Get State Statistics
        print("Last Traj: STD's: {}, Means: {}".format(path_stds, path_means))
        radii = (path_means + (path_stds * num_stds_per_step)) * num_steps_in_waypoint_radii  # radii for waypoints
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
        waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
        path_stds, path_means = path_deltas_stds_and_means_per_dim(path) # Get State Statistics
        print("Best Traj: STD's: {}, Means: {}".format(path_stds, path_means))
        radii = radii_calc(path_means, path_stds, num_means_per_step, num_stds_per_step, num_steps_in_waypoint_radii) # radii for waypoints


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





if __name__ == "__main__":
    first_num_episodes = None
    target_default_directory = "ddpg_baselines_summaries/1"

    files = """DDPG_Baselines_agent_MountainCarContinuous-v0-1000ep-NoGPU-noNorm-LLTanh-decayingNoise.json""".split()
    # summaries = []
    for file_name in files:
        target_file_name = file_name
        plot_file(target_default_directory, target_file_name, paths=True, first_num_episodes=first_num_episodes)
        target_path = os.path.join(get_default_data_directory(target_default_directory), target_file_name)
        summary = Summary.load(target_path)

        if summary.param_dict:
            print(file_name + ":")
            for key,value in sorted(summary.param_dict.items(), key=lambda x: x[0]):
                print("    " + key + " : " + str(value))
            print("")

    show_plot()