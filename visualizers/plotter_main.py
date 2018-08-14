import glob
import re

from smartstart.utilities.plot import plot_path, plot_summary, \
    mean_reward_episode, steps_episode, show_plot, total_rewards_episode, \
    plot_get_axes_and_fig, plot_set_legend_patches, plot_set_legend_lines
from smartstart.utilities.utilities import get_default_data_directory, get_start_waypoints_final_states_steps, \
    get_glob_summaries
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc

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


def plot_file_rewards(file_path_s, ma_window=1, dots=True, colors=None, linestyles=None, linewidth=.1):
    """
    :param target_default_directory: name of directory folder under the default data directory
    :param target_file_name: Name OR List of names,
                             Name(s) must be EXACT except for wildcards like asdf_*.json which will match asdf_0.json, asdf_1.json etc.
    :return: 4 axes and 1 figure (for future plotting)
    """
    title = "multiple graphs" if len(file_path_s) > 1 else os.path.basename(file_path_s[0])
    ########################################

    # Plot results
    ((ax1, ax2), (ax3, ax4)), fig = plot_get_axes_and_fig(2,2)
    colors, linestyles = colors.copy(), linestyles.copy()
    colors.reverse()
    linestyles.reverse()

    plot_summary(file_path_s, mean_reward_episode, ma_window=ma_window, title="Average Reward per Episode\n" + title,
                 axis=ax1, dots=dots, colors=colors.copy(), linestyles=linestyles.copy(), linewidth=linewidth)
    plot_summary(file_path_s, steps_episode, ma_window=ma_window, title="Steps per Episode\n" + title,
                 axis=ax2, dots=dots, colors=colors.copy(), linestyles=linestyles.copy(), linewidth=linewidth)
    plot_summary(file_path_s, total_rewards_episode, ma_window=ma_window, title="Total Reward per Episode\n" + title,
                 axis=ax3, dots=dots, colors=colors.copy(), linestyles=linestyles.copy(), linewidth=linewidth)
    fig.tight_layout()

    return ax1, ax2, ax3, ax4, fig

def plot_file_paths(summary):
    # Settings##############################
    # waypoint size
    linewidth = 3
    num_stds_per_step = 1
    num_means_per_step = 1
    num_steps_in_waypoint_radii = 1
    steps_between_waypoints = 5

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

def get_hidden_layer_label(main_summaries, show_seed=False):
    pd = main_summaries[0].param_dict
    retStr =  "a-" + str(pd['actor_h1']) + "-" + str(int(pd['actor_h2'])) + "-" + str(pd['actor_lr']) + "_" + \
              "c-" + str(pd['critic_h1']) + "-" + str(int(pd['critic_h2'])) + "-" + str(pd['critic_lr'])
    if show_seed:
        retStr += "_seed-" + str(pd['zz_RANDOM_SEED'])
    return retStr


def print_color_linestyle_and_file_name(colors, linestyles, file_path_s_batch):
    print("\nBatch Begin" + "#" * 150)
    for i in range(len(colors)):
        print(colors[i] + "," + linestyles[i] + " || " + os.path.basename(file_path_s_batch[i]))
    print("Batch END" + "#" * 150)

if __name__ == "__main__":
    first_num_episodes = None
    # target_default_directory = "ddpg_baselines_summaries/hidden_layer_size_experiment"
    target_default_directory = "ddpg_baselines_summaries/good_params"
    notSummary = True

#     files = """MountainCarContinuous-v0_a-64-32.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-64-32.0-0d001_c-128-64.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-64-32.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-128-64.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-128-64.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json
# MountainCarContinuous-v0_a-200-100.0-0d001_c-200-100.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json""".split()
#     files = """MountainCarContinuous-v0_a-64-32.0-0d001_c-64-32.0-0d001_1000ep_NoGPU_noNorm_LLTanh-decayingNoise_*.json""".split()
    files = """MountainCarContinuous-v0_a-64-32-0d001_c-64-32-0d001_*.json""".split()


    # files = [s for s in files if re.match("^.*0d001.*0d001.*$", s)]
    #COLOR PARAMETERS #################################################################################
    # blue, red, cyan = '001FFF', '#FF0000',
    colors_base = ['blue', 'red', 'green', 'brown', 'cyan', 'black']
    linestyles_base = ['solid', 'dotted']
    # list of colors and linestyles (rotate through colors first, then change linestyle, rotate again etc.)
    colors = colors_base * len(linestyles_base)
    linestyles = []
    for style in linestyles_base:
        linestyles.extend([style] * len(colors_base))



    # OTHER PARAMETERS ################################################################################
    ma_window = 1
    num_plots_per_graph = len(colors)
    plot_path_bool = False
    print_param_dict = False
    smart_start_dots = False
    average_over_glob_wildcards=True
    show_seed=False
    linewidth = 2


    # processing the file names into file paths, and ensuring the color/linstyle lists are the right length
    file_path_s_all = convert_file_names_to_file_paths(target_default_directory, files)
    if not average_over_glob_wildcards:
        new_file_path_s_all = []
        for fp in file_path_s_all:
            new_file_path_s_all.extend(glob.glob(fp)) #makes each glob'ed entry its own entry
        file_path_s_all = new_file_path_s_all

    num_plots_per_graph = min(num_plots_per_graph, len(file_path_s_all))
    colors, linestyles = colors[:num_plots_per_graph], linestyles[:num_plots_per_graph]
    #process the graph into chunks ('num_plots_per_graph' at a time)
    for file_path_s_batch in [file_path_s_all[i:i + num_plots_per_graph] for i in range(0, len(file_path_s_all), num_plots_per_graph)]:



        ax1, ax2, ax3, ax4, fig = plot_file_rewards(file_path_s_batch, ma_window=ma_window, dots=smart_start_dots,
                                                    colors=colors, linestyles=linestyles, linewidth=linewidth)

        labels = []
        for file_path in file_path_s_batch:
            main_summaries = get_glob_summaries(file_path) # loads all summary files found by glob

            # legend for labelling different files that go on the same graphs
            labels.append(get_hidden_layer_label(main_summaries, show_seed=show_seed))

            # plotting the best paths (and last path) on the first file found by glob
            if plot_path_bool:
                plot_file_paths(main_summaries[0])

            # printing out the parameter dictionary of the first file found by glob
            if print_param_dict and main_summaries[0].param_dict:
                print(os.path.basename(file_path) + ":")
                for key, value in sorted(main_summaries[0].param_dict.items(), key=lambda x: x[0]):
                    print("    " + key + " : " + str(value))
                print("")

        print_color_linestyle_and_file_name(colors, linestyles, file_path_s_batch)

        plot_set_legend_lines(colors, labels,linestyles=linestyles, axis=ax4)
        show_plot()
