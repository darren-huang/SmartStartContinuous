"""SARSA module

Module defining classes for SARSA and SARSA(lambda).

See 'Reinforcement Learning: An Introduction by Richard S. Sutton and
Andrew G. Barto for more information.
"""
from smartstart.RLDiscreteAlgorithms import TDLearning, TDLearningLambda


class SARSA(TDLearning):
    """

    Parameters
    ----------
    env : :obj:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning`
        environment
    *args :
        see parent :class:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning`
    **kwargs :
        see parent :class:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning`
    """

    def __init__(self, env, *args, **kwargs):
        super(SARSA, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            Next observation
        done : :obj:`bool`
            Boolean is True for terminal state

        Returns
        -------
        :obj:`float`
            Q-value for obs_tp1
        :obj:`int`
            action_tp1

        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1


class SARSALambda(TDLearningLambda):
    """

    Parameters
    ----------
    env : :obj:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning`
        environment
    *args :
        see parent :class:`~smartstart.RLDiscreteAlgorithms.tdlearning
        .TDLearningLambda`
    **kwargs :
        see parent :class:`~smartstart.RLDiscreteAlgorithms.tdlearning
        .TDLearningLambda`
    """

    def __init__(self, env, *args, **kwargs):
        super(SARSALambda, self).__init__(env, *args, **kwargs)

    def get_next_q_action(self, obs_tp1, done):
        """On-policy action selection

        Parameters
        ----------
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            Next observation
        done : :obj:`bool`
            Boolean is True for terminal state

        Returns
        -------
        :obj:`float`
            Q-value for obs_tp1
        :obj:`int`
            action_tp1

        """
        if not done:
            action_tp1 = self.get_action(obs_tp1)
            next_q_value = self.get_q_value(obs_tp1, action_tp1)
        else:
            next_q_value = 0.
            action_tp1 = None

        return next_q_value, action_tp1

#
# if __name__ == "__main__":
#     from smartstart.environments.gridworld import GridWorld, GridWorldVisualizer
#
#     directory = '/home/bartkeulen/repositories/smartstart/data/tmp'
#
#     np.random.seed()
#
#     env = GridWorld.generate(GridWorld.EXTREME)
#     visualizer = GridWorldVisualizer(env)
#     visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
#                               GridWorldVisualizer.CONSOLE,
#                               GridWorldVisualizer.VALUE_FUNCTION,
#                               GridWorldVisualizer.DENSITY)
#     env.visualizer = visualizer
#     # env.wall_reset = True
#
#     agent = SARSALambda(env, alpha=0.3, num_episodes=1000, max_steps=10000)
#
#     summary = agent.train(render=False, render_episode=True)
#
#     summary.save(directory=directory)

