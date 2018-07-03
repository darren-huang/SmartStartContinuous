"""Module for plotting results

This module describes methods for generating a few predefined plots. The
methods make it easy to plot multiple different experiments and average the
results of experiments with the same parameters.

The results are :class:`~smartstart.utilities.datacontainers.Summary` objects
saved as JSON strings.
"""
import glob
import os

import matplotlib.colorbar
import matplotlib.colors
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns

from smartstart.utilities.numerical import moving_average
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.utilities import get_start_waypoints_final_states

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = matplotlib.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def mean_reward_std_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot mean reward with standard deviation per episode

    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)
    std = np.std(rewards, axis=0)
    upper = moving_average(mean + std)
    lower = moving_average(mean - std)

    plt.fill_between(range(len(upper)), lower, upper, alpha=0.3, color=color)
    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def mean_reward_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot mean reward per episode

    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    rewards = np.array([np.array(summary.average_episode_reward()) for summary in summaries])

    mean = np.mean(rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


def steps_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot number of steps per episode


    Parameters
    ----------
    summaries : :obj:`list` of :obj:`~smartstart.utilities.datacontainers.Summary`
        summaries to average and plot
    ma_window : :obj:`int`
        moving average window size (Default value = 1)
    color :
        color (Default value = None)
    linestyle :
        linestyle (Default value = None)

    """
    steps = np.array([np.array(summary.steps_episode()) for summary in summaries])

    mean = np.mean(steps, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)

def total_rewards_episode(summaries, ma_window=1, color=None, linestyle=None):
    """Plot total_rewards

    """
    total_rewards = np.array([np.array(summary.total_episode_reward()) for summary in summaries])

    mean = np.mean(total_rewards, axis=0)
    ma_mean = moving_average(mean, ma_window)

    plt.plot(range(len(ma_mean)), ma_mean, color=color, linestyle=linestyle, linewidth=1.)


labels = {
    mean_reward_std_episode: ["Episode", "Average Reward"],
    mean_reward_episode: ["Episode", "Average Reward"],
    steps_episode: ["Episode", "Steps per Episode"],
    total_rewards_episode: ["Episode", "Total Reward"]
}

def plot_get_axes(ncols, nrows, fig_size_x=14, fig_size_y=10):
    """
    :param ncols:
    :param nrows:
    :return: ((ax1, ax2), (ax3, ax4)) if nrows=2,ncols=2
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size_x, fig_size_y))
    return axes



def plot_summary(files, plot_type, ma_window=1, title=None, legend=None,
                 output_dir=None, colors=None, linestyles=None,
                 format="eps", baseline=None, axis=None):
    """Main plot function to be used

    The files parameter can be a list of files or a list of
    :obj:`~smartstart.utilities.datacontainers.Summary` objects that you
    want to compare in a single plot. A single file or single
    :obj:`~smartstart.utilities.datacontainers.Summary` can also be provided.
    Please read the instructions below when supplying a list of files.

    The files list provided must contain filenames without the ``.json``
    extension. For example: ``['file/path/to/experiment']`` is correct but ``[
    'file/path/to/experiment.json']`` not! The reason for this is when the
    folder contains multiple summary files from the same experiment (same
    parameters) it will use all the files and average them. For example when
    the folder contains the following three files ``[
    'file/path/to/experiment_1.json', 'file/path/to/experiment_2.json',
    'file/path/to/experiment_3.json']``. By providing ``[
    'file/path/to/experiment']`` all three summaries will be loaded and averaged.

    Note:
        The entries in files have to be defined without ``.json`` at the end.

    Note:
        Don't forget to run the show_plot() function after initializing the
        plots. Else nothing will be rendered on screen

    Parameters
    ----------
    files : :obj:`list` of :obj:`str` or :obj:`list` of
    :obj:`~smartstart.utilities.datacontainers.Summary`
        Option 1: each entry is the filepath to a saved summary without
        ``.json`` at the end. Option 2: each entry is a Summary object.
    plot_type :
        one of the plot functions defined in this module
    ma_window : :obj:`int`
        moving average filter window size (Default value = 10)
    title : :obj:`str`
        title of the plot, is also used as filename (Default value = None)
    legend : :obj:`list` of :obj:`str`
        one entry per entry in files (Default value = None)
    output_dir : :obj:`str`
        if not None the plot will be saved in this directory (Default value =
        None)
    colors : :obj:`list`
        one entry per entry in files (Default value = None)
    linestyles : :obj:`list`
        one entry per entry in files (Default value = None)
    format : :obj:`str`
        output format when saving plot (Default value = "eps")
    baseline : :obj:`float`
        plotting a dotted horizontal line as baseline (Default value = None)
    """
    if colors is not None:
        assert len(colors) == len(files)
    if linestyles is not None:
        assert len(linestyles) == len(files)
    if type(files) is not list:
        files = [files]

    if axis:
        plt.axes(axis)
    else:
        plt.figure()

    xmax = 0
    for file in files:
        if type(file) is Summary:
            summaries = [file]
        else:
            fps = glob.glob("%s*.json" % file)
            summaries = [Summary.load(fp) for fp in fps]

        xmax = max(xmax, len(summaries[0]))

        color, linestyle = None, None
        if colors is not None:
            color = colors.pop()
        if linestyles is not None:
            linestyle = linestyles.pop()

        plot_type(summaries, ma_window, color, linestyle)

    if baseline is not None:
        plt.hlines(y=baseline, xmin=0, xmax=xmax, color="black", linestyle="dotted")

    if title is not None and output_dir is None:
        plt.title(title)
    plt.autoscale(enable=True, axis='x', tight=True)
    x_label, y_label = labels[plot_type]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(legend)

    if output_dir:
        save_plot(output_dir, title, format)

    plt.tight_layout()




def save_plot(output_dir, title, format="eps"):
    """Helper method for saving plots

    Parameters
    ----------
    output_dir : :obj:`str`
        directory where the plot is saved
    title : :obj:`str`
        filename of saved plot
    format : :obj:`str`
        file format (Default value = "eps")

    Raises
    ------
    Exception
        Please give a title when saving a figure.
    """
    sns.set_context("paper")
    if title is None:
        raise Exception("Please give a title when saving a figure.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = title.replace(" ", "_")
    fp = os.path.join(output_dir, filename + "." + format)
    plt.savefig(fp,
                format=format,
                dpi=1200,
                bbox_inches="tight")

def plot_path(path, title="", reward=None, x_label=None, y_label=None, num_waypoints=0, radii=[0, 0], linewidth=3):
    assert len(path[0]) == 2

    if reward is not None:
        title += " | Total Reward: {0:.2f}".format(reward)
    title += " | Steps: " + str(len(path) - 1)

    # a plot coloring the curve using a continuous colormap
    waypoint_collection = None

    # get x's and y's
    x = [s[0] for s in path]
    y = [s[1] for s in path]

    # color map variables to configure a gradient along the trajectory
    color_num_scale = np.linspace(0.0, len(path), len(path))
    cmap = plt.get_cmap('gnuplot2_r')
    norm = plt.Normalize(0, len(path))

    #draw trajectory
    line_collection = make_line_collection(x, y, color_num_scale, cmap=cmap, norm=norm, linewidth=linewidth)

    #maybe draw waypoints along trajectory
    if num_waypoints:
        waypoint_centers = get_start_waypoints_final_states(path, num_waypoints)
        waypoint_cmap = plt.get_cmap('binary')
        waypoint_color_num_scale = np.linspace(.2, .6, len(waypoint_centers))
        waypoint_norm = plt.Normalize(0, 1.0)
        waypoint_collection = make_ellipse_collection(waypoint_centers, waypoint_color_num_scale,
                                                      x_radius=radii[0], y_radius=radii[1],
                                                      cmap=waypoint_cmap, norm=waypoint_norm)
    #place stuff on the actual plot
    figure, (axis0, axis1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[12,1]})  # type: (object, (matplotlib.axes.Axes, matplotlib.axes.Axes))
    axis0.set_title(title)
    axis0.set_xlabel(x_label)
    axis0.set_ylabel(y_label)
    # plt.plot([state[0] for state in best_path], [state[1] for state in best_path], linewidth=1)
    if num_waypoints:
        axis0.add_collection(waypoint_collection)
    axis0.add_collection(line_collection)

    axis0.set_xlim(min(x) - radii[0], max(x) + radii[0])
    axis0.set_ylim(min(y) - radii[1], max(y) + radii[1])

    cb1 = matplotlib.colorbar.ColorbarBase(axis1, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Step')


def make_circle_collection(circle_centers, color_num_scale, radius=5,
                           cmap=plt.get_cmap('gnuplot2_r'),
                           norm=plt.Normalize(0, 1.0)):
    p = PatchCollection([Circle((c[0], c[1]), radius) for c in circle_centers],
                        cmap=cmap, norm=norm)
    p.set_array(color_num_scale)
    return p

def make_ellipse_collection(ellipse_centers, color_num_scale, x_radius=1, y_radius=1, angle=0,
                           cmap=plt.get_cmap('gnuplot2_r'),
                           norm=plt.Normalize(0, 1.0)):
    p = PatchCollection([Ellipse((c[0], c[1]), x_radius * 2, y_radius * 2, angle) for c in ellipse_centers],
                        cmap=cmap, norm=norm)
    p.set_array(color_num_scale)
    return p

def make_line_collection(x, y, color_num_scale,
                         cmap=plt.get_cmap('gnuplot2_r'),
                         norm =plt.Normalize(0, 1.0), linewidth=3):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    color_num_scale = np.asarray(color_num_scale)
    lc = LineCollection(segments, cmap=cmap,
                        norm=norm)
    lc.set_array(color_num_scale)
    lc.set_linewidth(linewidth)
    return lc


def show_plot():
    """Render the plots on screen

    Must be run after initializing the plots to actually show them on screen.
    """
    plt.show()