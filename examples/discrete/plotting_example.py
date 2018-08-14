import os

from smartstart.utilities.plot import plot_summary, \
    mean_reward_std_episode, steps_episode, show_plot
from smartstart.utilities.utilities import get_default_data_directory

# Get directory where the summaries are saved. Since it is the same folder as
#  the experimenter we can use the get_data_directory method
summary_dir = get_default_data_directory("")

# Define the files list
files = [os.path.join(summary_dir, "QLearning_GridWorldMedium"),
         os.path.join(summary_dir, "SmartStart_QLearning_GridWorldMedium")]

legend = ["Q-Learning", "SmartStart Q-Learning"]

# We are going to save the plots in img folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

# Plot average reward and standard deviation per episode
# When an output directory is supplied the plots will not be rendered with
# a title. The title is used as filename for the plot.
plot_summary(files, mean_reward_std_episode, ma_window=5, title="Q-Learning GridWorldMedium Average Reward per Episode",
             legend=legend, output_dir=output_dir)

plot_summary(files, steps_episode, ma_window=5, title="Q-Learning GridWorldMedium Steps per Episode", legend=legend,
             output_dir=output_dir, format=".png")

show_plot()