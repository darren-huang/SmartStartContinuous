import glob
import re

from smartstart.utilities.plot import plot_path, plot_summary, \
    mean_reward_episode, steps_episode, show_plot, total_rewards_episode, \
    plot_get_axes_and_fig, plot_set_legend_patches, plot_set_legend_lines
from smartstart.utilities.utilities import get_default_data_directory, get_start_waypoints_final_states_steps, \
    get_glob_summaries
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc, path_shortcutter, \
    elliptical_euclidean_distance_function_generator

import os

def convert_file_names_to_file_paths(target_default_directory, target_file_name_s):
    """
    given file names and a directory, generates the full paths to that file
    :param target_default_directory: a string
    :param file_name_s: either a string or list of strings
    :return: list of strings
    """
    if isinstance(target_file_name_s, list):
        file_path_s = [os.path.join(get_default_data_directory(target_default_directory), file_name) for file_name in target_file_name_s]
    else:
        file_path_s = [os.path.join(get_default_data_directory(target_default_directory), target_file_name_s[0])]
    return file_path_s


def plot_file_paths(summary, scale_up, plot_best_path=True, last_index =0, shorten_compare=True, theta=1):
    # Settings##############################
    # waypoint size
    linewidth = 3
    num_stds_per_step = 1
    num_means_per_step = 1
    num_steps_in_waypoint_radii = 1
    steps_between_waypoints = 5

    if last_index is not None:
        #LAST Trajectory##   ########################################################
        # which "last" trajectory to show
        path, reward, title = summary.get_last_path(last_index), summary.get_last_reward(last_index), str(last_index) + " From Last Path"
        waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
        path_stds, path_means = path_deltas_stds_and_means_per_dim(path)  # Get State Statistics
        print("Last Traj: STD's: {}, Means: {}".format(path_stds, path_means))
        radii = (path_means + (path_stds * num_stds_per_step)) * num_steps_in_waypoint_radii  # radii for waypoints
        if scale_up:
            radii = radii * (len(radii) ** .5)
        ## Plot Paths/Trajectories
        plot_path(path,
                  title=title,
                  reward=reward,
                  x_label="x_pos",
                  y_label="x_velocity",
                  waypoint_centers=waypoint_centers,
                  radii=radii,
                  linewidth=linewidth)
        if shorten_compare:
            ## Plot Paths/Trajectories
            distance_func = elliptical_euclidean_distance_function_generator(radii)
            path = path_shortcutter(path, distance_func, theta)
            waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
            plot_path(path,
                      title=title + "_shortcut",
                      reward=reward,
                      x_label="x_pos",
                      y_label="x_velocity",
                      waypoint_centers=waypoint_centers,
                      radii=radii,
                      linewidth=linewidth)

    if plot_best_path:
        #BEST Trajectory ##########################################################
        # number of waypoints for BEST PATH
        path, reward, title = summary.best_path, summary.best_reward, "Best Path"
        waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
        path_stds, path_means = path_deltas_stds_and_means_per_dim(path) # Get State Statistics
        print("Best Traj: STD's: {}, Means: {}".format(path_stds, path_means))
        radii = radii_calc(path_means, path_stds, num_means_per_step, num_stds_per_step, num_steps_in_waypoint_radii) # radii for waypoints
        if scale_up:
            radii = radii * (len(radii) ** .5)


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

        if shorten_compare:
            ## Plot Paths/Trajectories
            distance_func = elliptical_euclidean_distance_function_generator(radii)
            path = path_shortcutter(path, distance_func, theta)
            waypoint_centers = get_start_waypoints_final_states_steps(path, steps_between_waypoints)
            plot_path(path,
                      title=title + "_shortcut",
                      reward=reward,
                      x_label="x_pos",
                      y_label="x_velocity",
                      waypoint_centers=waypoint_centers,
                      radii=radii,
                      linewidth=linewidth)


if __name__ == "__main__":
    first_num_episodes = None
    # target_default_directory = "ddpg_baselines_summaries/hidden_layer_size_experiment"
    target_default_directory = "0_ddpg_summaries_DEPRACATED"
    notSummary = True

#     files = """MountainCarContinuous-v0_a-64-32.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-64-32.0-0d001_c-128-64.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-64-32.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-128-64.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-200-100.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json""".split()
#     target_file_names = """MountainCarContinuous-v0_a-64-32.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json""".split()
    # target_file_names = "DDPG_agent_MountainCarContinuous-v0-1000ep.json".split()
    target_file_names = "DDPG_agent_MountainCarContinuous-v0_test-2ep.json".split()




    # OTHER PARAMETERS ################################################################################
    print_param_dict = True
    average_over_glob_wildcards=False
    scale_up = False
    plot_best_path = True
    last_index = 0
    shorten_compare = True
    theta = 1


    # processing the file names into file paths, and ensuring the color/linstyle lists are the right length
    file_path_s_all = convert_file_names_to_file_paths(target_default_directory, target_file_names)
    if not average_over_glob_wildcards:
        new_file_path_s_all = []
        for fp in file_path_s_all:
            new_file_path_s_all.extend(glob.glob(fp)) #makes each glob'ed entry its own entry
        file_path_s_all = new_file_path_s_all

    for file_path in file_path_s_all:
        main_summaries = get_glob_summaries(file_path) # loads all summary files found by glob
        # plotting the best paths (and last path) on the first file found by glob
        plot_file_paths(main_summaries[0], scale_up, plot_best_path=plot_best_path,
                        last_index=last_index, shorten_compare=shorten_compare, theta=theta)
        # printing out the parameter dictionary of the first file found by glob
        if print_param_dict and main_summaries[0].param_dict:
            print(os.path.basename(file_path) + ":")
            for key, value in sorted(main_summaries[0].param_dict.items(), key=lambda x: x[0]):
                print("    " + key + " : " + str(value))
            print("")
    show_plot()
