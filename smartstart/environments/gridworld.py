"""GridWorld module

"""
import math
from collections import defaultdict

import numpy as np
from gym.envs.registration import EnvSpec

from smartstart.environments.environment import Environment
from smartstart.environments.generate_gridworld import generate_gridworld
from smartstart.environments.presets import *


class GridWorld(Environment):
    """GridWorld environment

    Creates a GridWorld environment with the layout specified by the user. A
    layout is is double numpy array, where each entry is a state with a
    certain type. The possible types are:

        0.   State
        1.   Wall
        2.   Start state
        3.   Goal state

    Example layout = [[3, 0, 0],[1, 1, 0],[1, 1, 0],[2, 0, 0]]

    The agent can take discrete steps in 4 directions; up, right, down and
    left. When no transition probability is defined (T_prob=0) the
    environment will be deterministic (i.e. action up will bring the agent to
    state above the current state). When a transition probability is defined
    there is T_prob change the agent will choose a random action from the
    four actions.

    When the agent tries to step through the wall it will count as a step but
    the agent will stay in the same state. When wall_reset is set to True the
    environment will return done to the learning algorithm.

    A reward of 1. is given for reaching the goal state and the environment will
    return done=True to the learning algorithm.

    Five different preset gridworlds are defined; EASY, MEDIUM, HARD, EXTREME
    and IMPOSSIBRUUHHH. The first three have a simple layout like,
    the EXTREME maze is a more complicated maze and the IMPOSSIBRUUHHH
    generates a random maze with a specified size.

    Parameters
    ----------
    name : :obj:`str`
        name for the environment, class name will be prefixed to this name
    layout : double :obj:`list` of `int` or `np.ndarray`
        layout of the gridworld
    T_prob : :obj:`float`
        transition probability
    wall_reset : :obj:`bool`
        when True, the environment will return done when a wall is hit
    scale : :obj:`int`
        scale factor, gridworld will become layout * scale in size
    """
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3
    IMPOSSIBRUUHHH = 4

    def __init__(self, name, layout, T_prob=0., wall_reset=False, scale=5):
        super(GridWorld, self).__init__(self.__class__.__name__ + name)
        layout = np.asarray(layout)
        self.T_prob = T_prob

        grid_world = np.kron(layout, np.ones((scale, scale), dtype=layout.dtype))
        start_state = np.asarray(np.where(grid_world == 2))[:, math.floor(scale**2/2)]
        goal_state = np.asarray(np.where(grid_world == 3))[:, math.floor(scale**2/2)]
        grid_world[grid_world == 2] = 0
        grid_world[grid_world == 3] = 0
        grid_world[tuple(start_state)] = 2
        grid_world[tuple(goal_state)] = 3

        self.h, self.w = grid_world.shape
        self.num_actions = 4
        self.grid_world, self.start_state, self.goal_state = grid_world, start_state, goal_state

        self.wall_reset = wall_reset

        self.spec = EnvSpec(self.__class__.__name__ + name+"-v0")

    def get_all_states(self):
        """Return all the states of the gridworld

        Returns
        -------
        :obj:`set`
            states
        """
        h, w = self.grid_world.shape
        states = set()
        for y in range(h):
            for x in range(w):
                if self.grid_world[y, x] != 1:
                    states.add((y, x))
        return states

    def get_T_R(self):
        """Return transition model and reward function

        Creates a the transition model and reward function for the gridworld
        and transition probability. Can be used by dynamic programming, like
        :class:`~smartstart.RLDiscreteAlgorithms.valueiteration.ValueIteration`.

        Returns
        -------
        :obj:`collections.defaultdict`
            transition model
        :obj:`collections.defaultdict`
            reward function

        """
        states = self.get_all_states()

        T = defaultdict(lambda: defaultdict(lambda: 0))
        R = defaultdict(lambda: defaultdict(lambda: 0))
        for state in states:
            cur_state = np.asarray(state)
            for action in self.possible_actions(cur_state):
                for p_action in self.possible_actions(cur_state):
                    new_state = self._move(cur_state.copy(), p_action)
                    if (new_state < 0).any() or (new_state[0] >= self.h) or (new_state[1] >= self.w) or (
                        self.grid_world[tuple(new_state)] == 1):
                        new_state = cur_state.copy()

                    if self.grid_world[tuple(new_state)] == 3:
                        R[tuple(cur_state) + (action,)][tuple(new_state)] = 1.

                    if action == p_action:
                        p = 1 - self.T_prob + self.T_prob / len(self.possible_actions(cur_state))
                    else:
                        p = self.T_prob / len(self.possible_actions(cur_state))

                    T[tuple(cur_state) + (action,)][tuple(new_state)] += p

        return T, R

    def reset(self, start_state=None):
        """

        Parameters
        ----------
        start_state : :obj:`np.ndarray`, optional
             (Default value = None)

        Returns
        -------
        :obj:`np.ndarray`
            state of the agent (start state)
        """
        if start_state is not None:
            self.state = start_state
        else:
            self.state = self.start_state.copy()
        return self.state

    def _move(self, state, action):
        """Moves the agent according to action

        Parameters
        ----------
        state : :obj:`np.ndarray`
            state to move from
        action : :obj:`np.ndarray`
            action to execute

        Returns
        -------
        :obj:`np.ndarray`:
            next state

        Raises
        ------
        NotImplementedError
            when an invalid action is chosen
        """
        if action == 0:
            movement = np.array([0, 1])
        elif action == 1:
            movement = np.array([1, 0])
        elif action == 2:
            movement = np.array([0, -1])
        elif action == 3:
            movement = np.array([-1, 0])
        else:
            raise NotImplementedError("Invalid action %d. Available actions: [0, 1, 2, 3]")

        return state + movement

    def step(self, action):
        """Take a step in the environment

        Executes action and retrieves the new state. Determines if any wall
        is hit and if so goes back to previous state.

        Calculates the reward using the previous state, action and new state.
        Determines if the new state is terminal.

        Parameters
        ----------
        action : :obj:`int`
            action

        Returns
        -------
        :obj:`np.ndarray`
            new state
        :obj:`float`
            reward
        :obj:`bool`
            True if new state is terminal
        :obj:`dict`
            Empty info dict (to make it equal to the step method in the OpenAI
            Gym)
        """
        if np.random.rand() < self.T_prob:
            action = np.random.choice(self.possible_actions(self.state))

        new_state = self._move(self.state, action)

        if (new_state < 0).any() or (new_state[0] >= self.h) or (new_state[1] >= self.w) or (self.grid_world[tuple(new_state)] == 1):
            if self.wall_reset:
                return self.state, 0., True, {}
            else:
                new_state = self.state.copy()

        r = 1. if np.array_equal(new_state, self.goal_state) else 0.
        done = True if np.array_equal(new_state, self.goal_state) else False

        self.state = new_state

        return self.state, r, done, {}

    def possible_actions(self, state):
        """Returns the available actions for state

        Note:
            State is not used in this environment, might be necessary for
            more complicated environments where states don't have the same
            set actions.

        Parameters
        ----------
        state : :obj:`np.ndarray`
            state

        Returns
        -------
        :obj:`list` of :obj:`int`
            containing available actions for state
        """
        return [0, 1, 2, 3]

    def get_grid(self):
        """Returns copy of the gridworld

        Returns
        -------
        :obj:`np.ndarray`
            Copy of the gridworld
        """
        grid_copy = self.grid_world.copy()
        return grid_copy

    def close_render(self):
        """Close the visualizer

        Returns
        -------
        :obj:`bool`
            True if the visualizer was closed properly
        """
        return self.visualizer.render(close=True)

    def render(self, **kwargs):
        """Render the environment

        If no visualizer is defined prints it to the console.

        Parameters
        ----------
        **kwargs :
            See :class:`~smartstart.environments.gridworldvisualizer
            .GridWorldVisualizer` for details

        Returns
        -------
        :obj:`bool`
            True if the agent has to stop rendering
        """
        if self._visualizer is None:
            print("No visualizer attached")
            return True
        return self.visualizer.render(**kwargs)

    @classmethod
    def generate(cls, type=EASY, size=None):
        """Generates a gridworld according to the presets

        Preset are given in :mod:`smartstart.environments.presets`.
        Presets contain the name, layout and scale of the gridworld.

        Parameters
        ----------
        type : :obj:`int`
             gridworld type as defined in the class attributes (Default
             value = EASY)
        size : :obj:`tuple`, optional
             used for generating a random gridworld when the type is
             IMPOSSIBRUUHH.

        Returns
        -------
        :obj:`~smartstart.environments.environment.gridworld.GridWorld`
            new gridworld environment
        """
        if type == GridWorld.EASY:
            name, layout, scale = easy()
        elif type == GridWorld.MEDIUM:
            name, layout, scale = medium()
        elif type == GridWorld.HARD:
            name, layout, scale = hard()
        elif type == GridWorld.EXTREME:
            name, layout, scale = extreme()
        elif type == GridWorld.IMPOSSIBRUUHHH:
            name, layout, scale = generate_gridworld(size=size)
        else:
            raise NotImplementedError("Please choose from the available GridWorld implementations or build one your self.")

        return cls(name, layout, scale=scale)


# if __name__ == "__main__":
#     import time
#
#     env = GridWorld.generate(GridWorld.MEDIUM)
#     vis = GridWorldVisualizer(env)
#     vis.add_visualizer(GridWorldVisualizer.LIVE_AGENT)
#     env.visualizer = vis
#
#     obs = env.reset()
#
#     steps = []
#     num_episodes = 50
#     reached_goal = False
#     i_episode = 0
#     render = True
#     while not reached_goal:
#         obs = env.reset()
#         start = time.time()
#         i_step = 0
#         while True:
#             action = np.random.randint(4)
#
#             if render:
#                 render = env.render()
#
#             obs_tp1, r, done, _ = env.step(action)
#
#             obs = obs_tp1
#             i_step += 1
#
#             # if i_step % 10000 == 0:
#             #     elapsed = time.time() - start
#             #     print("%d steps in %.2f seconds" % (i_step, elapsed))
#
#             if done:
#                 if r > 0:
#                     reached_goal = True
#                 break
#
#         elapsed = time.time() - start
#         print("Episode %d finished in %d steps and %.2f seconds" % (i_episode, i_step, elapsed))
#         steps.append(i_step)
#         i_episode += 1
#
#     print("")
#     print("GOAL REACHED IN %d EPISODES" % i_episode)
#     print("Average number of steps over %d episodes is %.f" % (num_episodes, sum(steps) / len(steps)))