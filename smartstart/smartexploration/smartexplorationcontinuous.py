"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object. The
SmartStart object will be a subclass of the original algorithm object.
"""
import random
from queue import Queue
from typing import Deque

import numpy as np

from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.RLDiscreteAlgorithms import ValueIteration
from smartstart.utilities.datacontainers import Episode, Summary
from smartstart.reinforcementLearningCore.agents import RLAgent, ValueFuncRLAgent


class SmartStartContinuous(RLAgent):
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
    agent = ...  # type: ValueFuncRLAgent

    def K(self, a):  # Gaussian Kernel with sigma = 1 // not normalized
        return np.exp((-.5) * (a ** 2) / (self.sigma ** 2))

    def __init__(self,
                 agent,
                 env,
                 exploitation_param=1.,
                 exploration_param=2.,
                 eta=0.5,
                 m=1,
                 vi_gamma=0.99,
                 vi_min_error=1e-5,
                 vi_max_itr=1000,
                 buffer_size=10000,
                 n_ss = 1000,
                 sigma = 1):
        self.__class__.__name__ = "SmartStart_" + agent.__class__.__name__
        self.exploitation_param = exploitation_param
        self.exploration_param = exploration_param
        self.eta = eta
        self.m = m
        self.sigma = sigma

        # objects to interact with
        self.agent = agent
        self.env = env

        # SmartStart Trajectory Optimization

        # keeps track of some distribution of all states visited
        #TODO CHANGE BUFFER
        self.replay_buffer = ReplayBuffer(self, buffer_size) # FIFO replacement strategy

        self.n_ss = n_ss # number of states in buffer to consider for being Smart Start State


        # keep track of SmartStartPathing vs. NormalAgentPathing
        self.smart_start_pathing = False
        self.smart_start_path = None  # placeholder for smart_start_state to navigate to


    @property
    def normal_agent_pathing(self):
        return not self.smart_start_pathing

    def get_smart_start_path(self):
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
        if not self.replay_buffer: # if buffer is emtpy, nothing to evaluate
            return None

        possible_start_indices = self.replay_buffer.get_possible_smart_start_indices(self.n_ss)

        smart_start_index = None
        max_ucb = -float('inf')

        #find the smart_start state
        for main_step_index in possible_start_indices:
            # state value
            main_step = self.replay_buffer.buffer[main_step_index]
            state = self.replay_buffer.step_to_s2(main_step)
            state_value = self.agent.get_state_value(state)

            #C_hat calculation###############
            C_hat = 0

            #bandwith
            h = ((4 / (3 * len(self.replay_buffer))) ** (1 / 5)) * self.sigma #silverman's rule of thumb

            #iterate over all states in replay buffer, calculated the kernel density estimation
            other_state = self.replay_buffer.step_to_s(self.replay_buffer.buffer[0])
            C_hat += self.K((np.linalg.norm(state - other_state)) / h)
            for other_step in self.replay_buffer.buffer:
                other_state = self.replay_buffer.step_to_s2(other_step)
                C_hat += self.K((np.linalg.norm(state - other_state)) / h)
            C_hat = C_hat / h

            #ucb calculation
            ucb = self.exploitation_param * state_value + \
                  np.sqrt((self.exploration_param *
                           np.log(len(self.replay_buffer))) / C_hat)
            if ucb > max_ucb:
                smart_start_index = main_step_index
                max_ucb = ucb

        return self.replay_buffer.get_episodic_path_to_buffer_index(smart_start_index)

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
        # new_state added (maybe) to buffer
        self.state_to_buffer(new_state)

        # agent observation
        self.agent.observe(state, action, reward, new_state, done)

        # check if smart_start_state has been reached
        if self.smart_start_pathing and np.array_equal(new_state, self.smart_start_path):
            self.smart_start_pathing = False

    def start_new_episode(self, state):
        # new_state added (maybe) to buffer
        self.state_to_buffer(state)

        self.agent.start_new_episode(state)
        if np.random.rand() <= self.eta: #eta is probability of using smartStart
            self.smart_start_path = self.get_smart_start_path() # new state to navigate to

            #TODO: finish everything after this comment
            if self.smart_start_path is not None and not np.array_equal(self.smart_start_path, state): #valid smart start
                self.smart_start_pathing = True

                self.dynamic_programming(self.smart_start_path) #this will update the policy


    def render(self, env, **kwargs):
        kwargs['smart_start_state_density_map'] = self.get_smart_state_density_map()
        return self.agent.render(env, **kwargs)

    #private method, adds (maybe -rnd) state to buffer
    def state_to_buffer(self, new_state):
        # TODO CHANGE BUFFER
        if len(self.replay_buffer) < self.buffer_size: # still has capacity
            self.replay_buffer.append(new_state)
        else: # buffer full -> random replacement
            index = random.randint(0, self.buffer_size)
            if index != self.buffer_size:
                self.replay_buffer[index] = new_state


if __name__ == "__main__":
    from smartstart.environments.gridworld import GridWorld
    from smartstart.environments.gridworldvisualizer import GridWorldVisualizer
    from smartstart.RLDiscreteAlgorithms import SARSALambda

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    env = GridWorld.generate(GridWorld.EASY)
    visualizer = GridWorldVisualizer(env)
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env.visualizer = visualizer
    # env.T_prob = 0.1

    # env.wall_reset = True

    # agent = generate_smartstart_object(SARSALambda, env, eta=0.75, alpha=0.3,
    #                                    num_episodes=1000, max_steps=1000,
    #                                    exploitation_param=0.)
    agent = SmartStartDiscrete(SARSALambda(env, ta=0.75, alpha=0.3,
                                       num_episodes=1000, max_steps=1000), env,
                                       exploitation_param=0.)

    summary = agent.train(render=False, render_episode=True)

    summary.save(directory=directory)
