"""Counter module

Describes Counter base class for TD-learning RLDiscreteAlgorithms
"""
from collections import defaultdict

import numpy as np


class Counter(object):
    """Base class for visitation counts.
    
    Base class for keeping track of obs-action-obs_tp1 visitation counts in
    discrete state and discrete action reinforcement learning RLDiscreteAlgorithms.

    Parameters
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment

    Attributes
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    count_map : :obj:`collections.defaultdict` (nested)
        visitation counts for each obs-action-obs_tp1
    total : :obj:`int`
        total number of visitation counts

    """

    def __init__(self, env):
        self.env = env
        self.count_map = defaultdict(lambda:
                                     defaultdict(lambda:
                                                 defaultdict(lambda: 0)))
        self.total = 0

    def increment(self, obs, action, obs_tp1):
        """Increment count for obs-action-obs_tp1 transition.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            current observation
        action : :obj:`int`
            current action
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation

        """
        self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)] += 1
        self.total += 1

    def get_count(self, obs, action=None, obs_tp1=None):
        """Returns visitation count
        
        Visitation count for obs, obs-action or obs-action-obs_tp1 is returned.
        User can leave action and/or obs_tp1 empty to return a higher level
        count.
        
        Note:
            When action is None and obs_tp1 is not None, the count for just obs
            will be returned and obs_tp1 will not be taken into account.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action (Default value = None)
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation (Default value = None)

        Returns
        -------
        :obj:`int`
            Visitation count

        """
        if action is None:
            total = 0
            for value in self.count_map[tuple(obs)].values():
                total += sum(value.values())
            return total
        if obs_tp1 is None:
            return sum(self.count_map[tuple(obs)][(action,)].values())
        else:
            return self.count_map[tuple(obs)][(action,)][tuple(obs_tp1)]

    def get_density(self, obs, action=None, obs_tp1=None):
        """Density for obs, obs-action or obs-action-obs_tp1 is returned.
        
        Density is calculated by dividing the count with the total count.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            observation
        action : :obj:`int`
            action (Default value = None)
        obs_tp1 : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
            next observation (Default value = None)

        Returns
        -------
        :obj:`float`
            Density

        """
        count = np.sum(self.get_count(obs, action, obs_tp1))
        return count / self.total

    def get_count_map(self):
        """Returns state count map for environment.
        
        Count-map will be a numpy array equal to the width (w) and height (h)
        of the environment. Each entry (state) will hold the count associated
        with that state.
        
        Note:
            Only works for 2D-environments that have a w and h attribute.

        Returns
        -------
        :obj:`np.ndarray`
            Count map

        """
        count_map = np.zeros((self.env.h, self.env.w), dtype=np.int)
        for i in range(self.env.h):
            for j in range(self.env.w):
                count_map[i, j] = self.get_count([i, j])
        return count_map

    def get_density_map(self):
        """Returns state density map for environment
        
        Density-map will be a numpy array equal to the width (w) and height (h)
        of the environment. Each entry (state) will hold the density
        associated with that state.
        
        Note:
            Only works for 2D-environments that have a w and h attribute.

        Returns
        -------
        :obj:`np.ndarray`
            Density map

        """
        count_map = self.get_count_map()
        if np.sum(count_map) == 0:
            return count_map
        return count_map / self.total
