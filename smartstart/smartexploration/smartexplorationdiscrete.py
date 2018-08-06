"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object. The
SmartStart object will be a subclass of the original algorithm object.
"""
import random

import numpy as np

from RLDiscreteAlgorithms import QLearning
from reinforcementLearningCore.rlTrain import rlTrain
from smartstart.RLDiscreteAlgorithms import ValueIteration
from smartstart.reinforcementLearningCore.agents_abstract_classes import RLAgent, ValueFuncAndCountRLAgent
from utilities.plot import plot_summary, show_plot, mean_reward_episode, steps_episode


class SmartStartDiscrete(RLAgent):
    """SmartStart

    SmartStart algorithm consists of two stages:

        1.  Select smart start
        2.  Guide to smart start

    The smart start is selected using the UCB1 algorithm,
    as described at by the get_start method.

    The guiding to the smart start is done using model-based
    reinforcement learning. A transition model is fit using the
    visitation counts. A reward function that is zero expect for
    transitions that directly transition to the smart start,
    those transitions get a reward of one.

    Subsequently this transition model and the reward function are
    used to find a optimal policy using dynamic programming. This
    implementation uses
    :class:`~smartstart.RLDiscreteAlgorithms.valueiteration.ValueIteration`.

    After reaching the smart start the agent will continue with the
    normal reinforcement learning as described by the base class.

    Parameters
    ----------
    agent : :obj:'~smartstart.reinforcementLearningCore.agents.ValueFuncAndCountMapRLAgent'
        agent that runs once smartstart pathing is over (must also be a Counter, and have a .get_state_value func)
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    exploitation_param : :obj:`float`
        scaling factor for value function in smart start selection
    exploration_param : :obj:`float`
        scaling factor for density in the smart start selection
    eta : :obj:`float` or :obj:~smartstart.utilities.scheduler.Scheduler`
        change of using smart start at the start of an episode or
        executed the normal algorithm
    m : :obj:`int`
        number of state-action visitation counts needed before the
        state-action pair is used in fitting the transition model.
    vi_gamma : :obj:`float`
        discount factor for value iteration
    vi_min_error : :obj:`float`
        minimum error for convergence of value iteration
    vi_max_itr : :obj:`int`
        maximum number of iteration of value iteration
    *args :
        see the base class for possible parameters
    **kwargs :
        see the base class for possible parameters

    Parameters
    ----------
    exploitation_param : :obj:`float`
        scaling factor for value function in smart start selection
    exploration_param : :obj:`float`
        scaling factor for density in the smart start selection
    eta : :obj:`float` or :obj:~smartstart.utilities.scheduler.Scheduler`
        change of using smart start at the start of an episode or
        executed the normal algorithm
    m : :obj:`int`
        number of state-action visitation counts needed before the
        state-action pair is used in fitting the transition model.
    policy: :obj:`~smartstart.RLDiscreteAlgorithms.valueiteration.ValueIteration`
        policy used for guiding to the smart start
    """
    agent = ...  # type: ValueFuncAndCountRLAgent

    def __init__(self,
                 agent,
                 env,
                 exploitation_param=1.,
                 exploration_param=2.,
                 eta=0.5,
                 m=1,
                 vi_gamma=0.99,
                 vi_min_error=1e-5,
                 vi_max_itr=1000):
        self.__class__.__name__ = "SmartStart_" + agent.__class__.__name__
        self.exploitation_param = exploitation_param
        self.exploration_param = exploration_param
        self.eta = eta
        self.m = m

        # objects to interact with
        self.agent = agent
        self.env = env

        # valueIteration object for pathing to smartStart
        self.policy = ValueIteration(self.env, vi_gamma, vi_min_error,
                                     vi_max_itr)

        # counting the spots that have been chosen as smart_states
        self.smart_state_count = np.zeros((self.env.h, self.env.w), dtype=np.int)

        # keep track of SmartStartPathing vs. NormalAgentPathing
        self.smart_start_pathing = False
        self.smart_start_state = None # placeholder for smart_start_state to navigate to

    def get_param_dict(self):
        return None

    @property
    def normal_agent_pathing(self):
        return not self.smart_start_pathing

    def get_start(self):
        """Determines the smart start state

        The smart start is determined using the UCB1 algorithm. The UCB1
        algorithm is a well known exploration strategy for multi-arm
        bandit problems. The smart start is chosen according to

        smart_start = \arg\max\limits_s\left(alpha * \max\limits_a Q(s,
        a) + \sqrt{\frac{beta * \log\sum\limits_{s'} C(s'}{C(s}} \right)

        Where
            * \alpha = exploitation_param
            * \beta  = exploration_param

        Returns
        -------
        :obj:`np.ndarray`
            smart start
        """
        count_map = self.agent.get_count_map()
        if count_map is None:
            return None
        possible_starts = np.asarray(np.where(count_map > 0))
        if not possible_starts.any():
            return None

        smart_start = None
        max_ucb = -float('inf')
        for i in range(possible_starts.shape[1]):
            obs = possible_starts[:, i]
            state_value = self.agent.get_state_value(obs)
            ucb = self.exploitation_param * state_value + \
                  np.sqrt((self.exploration_param *
                           np.log(np.sum(count_map))) / count_map[tuple(obs)])
            if ucb > max_ucb:
                smart_start = obs
                max_ucb = ucb
        return smart_start

    def get_smart_state_density_map(self):
        """Returns smart state density map for environment

        Density-map will be a numpy array equal to the width (w) and height (h)
        of the environment. Each entry (state) will hold the density
        associated with that smart state.

        Note:
            Only works for 2D-environments that have a w and h attribute.

        Returns
        -------
        :obj:`np.ndarray`
            Density map

        """
        count_map = self.smart_state_count
        total_sum = np.sum(count_map)
        if total_sum == 0:
            return count_map
        return count_map / total_sum

    def dynamic_programming(self, start_state):
        """Fits transition model, reward function and performs dynamic
        programming

        Transition model is fitted using the following equation

            T(s,a,s') = \frac{C(s,a,s'}{C(s,a)}

        Where C(*) is the visitation count. The reward function is zero
        everywhere except for the transition that results in the smart start

            R(s,a,s') = 1 if s' == s_{ss}
            R(s,a,s') = 0 otherwise

        Dynamic programming is done using value iteration.

        Parameters
        ----------
        start_state : :obj:`np.ndarray`
            SmartStart state

        """
        # Reset policy
        self.policy.reset()

        # Fit transition model and reward function
        for obs_c, obs_count in self.agent.count_map.items():
            for action, action_count in obs_count.items():
                for obs_tp1, count in action_count.items():
                    self.policy.add_obs(obs_c)
                    if obs_tp1 == tuple(start_state):
                        self.policy.R[obs_c + action][obs_tp1] = 1.
                    self.policy.T[obs_c + action][obs_tp1] = \
                        count / sum(self.agent.count_map[obs_c][action].
                                    values())

        # Perform dynamic programming
        self.policy.optimize()


    def get_action(self, state):
        if self.smart_start_pathing:
            return self.policy.get_action(state)
        else:
            return self.agent.get_action(state)

    def observe(self, state, action, reward, new_state, done):
        self.agent.observe(state, action, reward, new_state, done)

        if self.smart_start_pathing and np.array_equal(new_state, self.smart_start_state):
            self.smart_start_pathing = False


    def start_new_episode(self, state):
        self.agent.start_new_episode(state)
        if np.random.rand() <= self.eta: #eta is probability of using smartStart
            self.smart_start_state = self.get_start() # new state to navigate to
            if self.smart_start_state is not None: #valid smart start
                self.smart_start_pathing = True
                self.smart_state_count[self.smart_start_state[0]][self.smart_start_state[1]] += 1 # update self.smart_state_count

                self.dynamic_programming(self.smart_start_state) #this will update the policy


    def render(self, env, **kwargs):
        kwargs['smart_start_state_density_map'] = self.get_smart_state_density_map()
        return self.agent.render(env, **kwargs)


if __name__ == "__main__":
    from smartstart.environments.gridworld import GridWorld
    from smartstart.environments.gridworldvisualizer import GridWorldVisualizer

    # Reset the seed for random number generation
    random.seed()
    np.random.seed()

    # Create environment and visualizer
    grid_world = GridWorld.generate(GridWorld.MEDIUM)
    visualizer = GridWorldVisualizer(grid_world)
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY,
                              GridWorldVisualizer.SMART_STATE_DENSITY)

    # Initialize agent, see class for available parameters
    agent = QLearning(grid_world,
                      alpha=0.1,
                      epsilon=0.05,
                      exploration=QLearning.E_GREEDY)

    smartStartAgent = SmartStartDiscrete(agent, grid_world)

    # Train the agent, summary contains training data
    summary = rlTrain(smartStartAgent, grid_world,
                      render=True,
                      render_episode=True,
                      print_results=True,
                      num_episodes=500,
                      # num_episodes=50,
                      max_steps=1000)

    # Plot results
    plot_summary(summary, mean_reward_episode, ma_window=5,
                 title="Easy GridWorld Q-Learning Average Reward per Episode")
    plot_summary(summary, steps_episode, ma_window=5,
                 title="Easy GridWorld Q-Learning Steps per Episode")
    show_plot()
