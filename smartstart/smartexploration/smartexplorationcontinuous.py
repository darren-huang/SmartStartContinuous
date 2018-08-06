"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object. The
SmartStart object will be a subclass of the original algorithm object.
"""

import time

import numpy as np
import tensorflow as tf

from smartstart.RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from smartstart.RLAgents.NND_MB_agent import NND_MB_agent
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.reinforcementLearningCore.agents_abstract_classes import RLAgent, ValueFuncRLAgent, ReplayBufferRLAgent
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.utilities import get_default_data_directory, set_global_seeds


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
    agent : :obj:'~smartstart.reinforcementLearningCore.agents.ValueFuncRLAgent'
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
                 sess, #tf session
                 exploitation_param=1.,
                 exploration_param=2.,
                 eta=0.5,
                 m=1,
                 nnd_mb_run_num=None,
                 nnd_mb_steps_per_waypoint=1,
                 nnd_mb_mean_per_stepsize=1,
                 nnd_mb_std_per_stepsize=1,
                 nnd_mb_stepsizes_in_waypoint_radii=1,
                 nnd_mb_gamma=.75,
                 nnd_mb_horizontal_penalty_factor=.5,
                 nnd_mb_use_existing_training_data=True,
                 nnd_mb_horizon=20,
                 nnd_mb_num_control_samples=5000,
                 nnd_mb_num_episodes_for_aggregation=3,
                 buffer_size=500000,
                 n_ss = 1000,
                 sigma = 1):
        #TODO fix naming for smartStart - this affects class not object
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
        #TODO Document the necessity of having the base_agent inherit from ReplayBufferRLAgent
        if isinstance(agent, ReplayBufferRLAgent):
            self.replay_buffer = agent.replay_buffer
            agent.set_replay_buffer_main_agent(self)
        else:
            self.replay_buffer = ReplayBuffer(self, buffer_size)  # FIFO replacement strategy


        self.n_ss = n_ss # number of states in buffer to consider for being Smart Start State

        # keep track of SmartStartPathing vs. NormalAgentPathing
        self.smart_start_pathing = False
        self.smart_start_path = None  # placeholder for smart_start_state to navigate to

        # the agent for navigating to the smartstart
        self.nnd_mb_agent = NND_MB_agent(env, sess, replay_buffer=self.replay_buffer,
                                         steps_per_waypoint=nnd_mb_steps_per_waypoint,
                                         mean_per_stepsize=nnd_mb_mean_per_stepsize,
                                         std_per_stepsize=nnd_mb_std_per_stepsize,
                                         stepsizes_in_waypoint_radii=nnd_mb_stepsizes_in_waypoint_radii,
                                         gamma=nnd_mb_gamma, horizontal_penalty_factor=nnd_mb_horizontal_penalty_factor,
                                         save_dir_name=nnd_mb_run_num,
                                         use_existing_training_data=nnd_mb_use_existing_training_data,
                                         horizon=nnd_mb_horizon, num_control_samples=nnd_mb_num_control_samples,
                                         num_episodes_for_aggregation=nnd_mb_num_episodes_for_aggregation)

        self.times_for_smart_start = []

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
        if len(self.replay_buffer) == 0: # if buffer is emtpy, nothing to evaluate
            return None

        possible_start_indices = self.replay_buffer.get_possible_smart_start_indices(self.n_ss)

        if not possible_start_indices: # no valid states then return None
            return None

        smart_start_index = None
        max_ucb = -float('inf')

        #find the smart_start state TODO PARALLELEIZE WITH NUMPY
        for main_step_index in possible_start_indices:
            # state value
            main_step = self.replay_buffer.buffer[main_step_index]
            state = self.replay_buffer.step_to_s2(main_step)
            state_value = self.agent.get_state_value(state)

            #C_hat calculation###############
            #bandwith
            h = ((4 / (3 * len(self.replay_buffer))) ** (1 / 5)) * self.sigma #silverman's rule of thumb

            # #iterate over all states in replay buffer, calculated the kernel density estimation
            # other_state = self.replay_buffer.step_to_s(self.replay_buffer.buffer[0])
            # C_hat += self.K((np.linalg.norm(state - other_state)) / h)
            # for other_step in self.replay_buffer.buffer:
            #     other_state = self.replay_buffer.step_to_s2(other_step)
            #     C_hat += self.K((np.linalg.norm(state - other_state)) / h)
            all_states = self.replay_buffer.get_all_states()
            C_hat_temp_arr = self.K((np.linalg.norm(state - all_states, axis=1)) / h)
            C_hat = np.sum(C_hat_temp_arr) / h

            #ucb calculation
            ucb = self.exploitation_param * state_value + \
                  np.sqrt((self.exploration_param *
                           np.log(len(self.replay_buffer))) / C_hat)
            if ucb > max_ucb:
                smart_start_index = main_step_index
                max_ucb = ucb

        return self.replay_buffer.get_episodic_path_to_buffer_index(smart_start_index)

    def get_action(self, state):
        if self.smart_start_pathing:
            return self.nnd_mb_agent.get_action(state)
        else:
            return self.agent.get_action(state)

    def observe(self, state, action, reward, new_state, done):
        # new_state added (maybe) to buffer
        self.replay_buffer.add(self, state, action, reward, done, new_state)

        # agent observation
        self.agent.observe(state, action, reward, new_state, done)

        # check if smart_start_state has been reached
        if self.smart_start_pathing:
            self.nnd_mb_agent.observe(state, action, reward, new_state, done)
            if self.nnd_mb_agent.close_enough_to_goal(new_state):
                self.smart_start_pathing = False
                print("distance to goal: " + str(self.nnd_mb_agent.distance_function(new_state,self.smart_start_path[-1])))
                print("END OF SMART START STUFFS")

    def start_new_episode(self, state):
        #TO/DO REMOVE FOLLOWING THIS IS JUST FOR TESTING
        # if True:
        if np.random.rand() <= self.eta: #eta is probability of using smartStart

            #TODO remove the random plotting
            start_time = time.time()
            self.smart_start_path = self.get_smart_start_path() # new state to navigate to

            if self.smart_start_path: #ensure path exists
                #TODO: remove the floowing plotting
                end_time = time.time()
                elapsed_time = end_time - start_time
                # self.times_for_smart_start.append(elapsed_time)
                # if len(self.times_for_smart_start) % 10 == 0:
                #     plt.title("Times for Calculating Smart Start Path")
                #     plt.plot(self.times_for_smart_start)
                #     plt.show()
                print("Calculate Smart Start Path Time: " + str(elapsed_time), end='')


                print("\npath exists")
                # let neural network dynamics model based controller load the path
                self.nnd_mb_agent.start_new_episode_plan(state, self.smart_start_path)
                if not self.nnd_mb_agent.close_enough_to_goal(state): #ensure goal hasn't already been reached
                    self.smart_start_pathing = True #this start smart start navigation
                    print("SMART_START START!!!")
                    return "smart_start"

        self.agent.start_new_episode(state)
        self.replay_buffer.start_new_episode(self)


    def render(self, env, **kwargs):
        return env.render()

if __name__ == "__main__":
    import random
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    # Reset the seed for random number generation
    RANDOM_SEED = 1234
    set_global_seeds(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    with tf.Session() as sess:
        # Initialize agent, see class for available parameters
        base_agent = DDPG_Baselines_agent(env, sess, buffer_size=10000, num_train_iterations=1,
                                          num_steps_before_train=1, actor_lr=0.01, critic_lr=0.005)

        smart_start_agent = SmartStartContinuous(base_agent, env, sess,
                                                 buffer_size=10000,
                                                 nnd_mb_run_num=0,
                                                 nnd_mb_steps_per_waypoint=1,
                                                 nnd_mb_mean_per_stepsize=1,
                                                 nnd_mb_std_per_stepsize=1,
                                                 nnd_mb_stepsizes_in_waypoint_radii=1,
                                                 nnd_mb_gamma=.75,
                                                 nnd_mb_horizontal_penalty_factor=.5,
                                                 nnd_mb_use_existing_training_data=True,
                                                 nnd_mb_horizon=4,
                                                 nnd_mb_num_control_samples=5000,
                                                 nnd_mb_num_episodes_for_aggregation=5,
                                                 n_ss=2000)
        sess.graph.finalize()

        # Train the agent, summary contains training data
        summary = rlTrain(smart_start_agent, env, render=False,
                          render_episode=False,
                          print_results=True, num_episodes=1000)  # type: Summary

    summary.save(get_default_data_directory("smart_start_continuous_summaries"), extra_name_append="-1000ep-2000n_ss")
