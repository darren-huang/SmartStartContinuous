"""Environment module

Contains base classes for constructing environments and visualizers
"""

from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """Base class for constructing environments

    """

    def __init__(self, name=None):
        self.name = name
        self._visualizer = None

    @property
    def visualizer(self):
        """:obj:`~smartstart.environments.environment.Visualizer` :
        visualizer of the agent, value function, density and/or console.

        Sets the name of the visualizer to be equal to the environments name
        """
        return self._visualizer

    @visualizer.setter
    def visualizer(self, visualizer):
        self._visualizer = visualizer
        self._visualizer.name = self.name

    @abstractmethod
    def reset(self, start_state=None):
        """Reset the agent to the start state

        Parameters
        ----------
        start_state : :obj:`np.ndarray`, optional
             Start state, if not supplied standard start state is used

        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Take a step in the environment

        Parameters
        ----------
        action : :obj:`int`, :obj:`float` or :obj:`np.ndarray`
            action

        Raises
        ------
        NotImplementedError
            use a subclass of
            :class:`~smartstart.environments.environment.Environment` like
            :class:`~smartstart.environments.gridworld.GridWorld`.
        """
        raise NotImplementedError


class Visualizer(metaclass=ABCMeta):
    CONSOLE = 0
    LIVE_AGENT = 1
    VALUE_FUNCTION = 2
    DENSITY = 3
    SMART_STATE_DENSITY = 4

    @abstractmethod
    def add_visualizer(self, *args):
        """Adds a window to the visualizer

        Available windows can be found in the implemented class attributes

        Parameters
        ----------
        args : :obj:`list` of :obj:`int`
            visualizers to add. Possible visualizers are defined in the
            class attributes

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def render(self, value_map=None, density_map=None, message=None,
               close=False):
        """Render the current state of the training algorithm

        The render method should include all functionality to close the
        visualizer. When the close parameter is True the visualizer should be
        closed and send back to the learning algorithm to stop rendering.

        Parameters
        ----------
        value_map : :obj:`np.ndarray`, optional
            value function map
        density_map : :obj:`np.ndarray`, optional
            density map
        message : :obj:`str`: optional
            message to print to the console window
        close : :obj:`bool`
            True if the visualizer has to be closed

        Returns
        -------
        :obj:`bool`
            True if the agent has to keep rendering

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError