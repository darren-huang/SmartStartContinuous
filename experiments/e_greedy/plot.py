import os

import matplotlib.pyplot as plt
import seaborn as sns

from utilities.plot import plot_mean

from experiments.e_greedy.experiment import alphas, directory, num_exp, gammas, epsilons

mazes = ["GridWorldEasy", "GridWorldMedium"]
algo = "QLearning"

for maze in mazes:
    files = []
    legend = []
    for alpha in alphas:
        legend.append(r'alpha = %.2f' % alpha)
        files.append(os.path.join(directory, "%s_%s_alpha=%.2f_gamma=0.990_epsilon=0.05" % (algo, maze, alpha)))
    title = "%s %s alpha" % (algo, maze)
    plot_mean(files, num_exp=num_exp, title=title, legend=legend)

    files = []
    legend = []
    for gamma in gammas:
        legend.append(r'gamma = %.3f' % gamma)
        files.append(os.path.join(directory, "%s_%s_alpha=0.10_gamma=%.3f_epsilon=0.05" % (algo, maze, gamma)))
    title = "%s %s gamma" % (algo, maze)
    plot_mean(files, num_exp=num_exp, title=title, legend=legend)

    files = []
    legend = []
    for epsilon in epsilons:
        legend.append(r'$\epsilon$ = %.2f' % epsilon)
        files.append(os.path.join(directory, "%s_%s_alpha=0.10_gamma=0.990_epsilon=%.2f" % (algo, maze, epsilon)))
    title = r"%s %s $\epsilon$" % (algo, maze)
    plot_mean(files, num_exp=num_exp, title=title, legend=legend)


plt.show()