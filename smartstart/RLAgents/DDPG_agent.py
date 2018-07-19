from abc import ABCMeta

from smartstart.utilities.plot import plot_path
from smartstart.reinforcementLearningCore.agents import RLAgent
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.RLContinuousAlgorithms.DDPG_MountainCar.actor import ActorNetwork
from smartstart.RLContinuousAlgorithms.DDPG_MountainCar.critic import CriticNetwork
from smartstart.RLContinuousAlgorithms.DDPG_MountainCar.ou_noise import OUNoise
from smartstart.utilities.utilities import get_default_directory
from smartstart.utilities.datacontainers import Summary
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import numpy as np




class DDPG_agent(RLAgent, metaclass=ABCMeta):
    #TODO edit this description
    """Base class for temporal-difference methods

    Base class for temporal difference methods Q-Learning, SARSA,
    SARSA(lambda) and Q(lambda). Implements all common methods, specific
    methods to each algorithm have to be implemented in the child class.

    All the exploration methods are defined in this class and can be added by
    adding a new method that describes the exploration strategy. The
    exploration strategy must be added to the class attribute below and
    inserted in the get_action method.

    Currently 5 exploration methods are implemented; no exploration,
    epsilon-greedy, boltzmann, count-based and ucb. Another exploration
    method is optimism in the face of uncertainty which can be used by
    setting the init_q_value > 0.

    Parameters
    ----------
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    num_episodes : :obj:`int`
        number of episodes
    max_steps : :obj:`int`
        maximum number of steps per episode
    alpha : :obj:`float`
        learning step-size
    gamma : :obj:`float`
        discount factor
    init_q_value : :obj:`float`
        initial q-value
    exploration : :obj:`str`
        exploration strategy, see class attributes for available options
    epsilon : :obj:`float` or :obj:`Scheduler`
        epsilon-greedy parameter
    temp : :obj:`float`
        temperature parameter for Boltzmann exploration
    beta : :obj:`float`
        count-based exploration parameter

    Attributes
    -----------
    num_episodes : :obj:`int`
        number of episodes
    max_steps : :obj:`int`
        maximum number of steps per episode
    alpha : :obj:`float`
        learning step-size
    gamma : :obj:`float`
        discount factor
    init_q_value : :obj:`float`
        initial q-value
    Q : :obj:`np.ndarray`
        Numpy ndarray holding the q-values for all state-action pairs
    exploration : :obj:`str`
        exploration strategy, see class attributes for available options
    epsilon : :obj:`float` or :obj:`Scheduler`
        epsilon-greedy parameter
    temp : :obj:`float`
        temperature parameter for Boltzmann exploration
    beta : :obj:`float`
        count-based exploration parameter
    """

    def __init__(self,
                 env,
                 replay_buffer=None,
                 epochs=1000,
                 action_bound=None,
                 MINIBATCH_SIZE=40,
                 GAMMA=0.99,
                 epsilon=1.0,
                 min_epsilon=0.01,
                 BUFFER_SIZE=10000,
                 train_indicator=True,
                 ACTOR_LEARNING_RATE=0.0001, # Base learning rate for the Actor network
                 CRITIC_LEARNING_RATE=0.001, # Base learning rate for the Critic Network
                 TAU=0.001,  # Soft target update param
                 EXPLORE=70, # set the exploration stuffs?
                 render=False):
        self.train_indicator = train_indicator
        self.min_epsilon = min_epsilon
        self.EXPLORE = EXPLORE
        self.epsilon = epsilon
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.GAMMA = GAMMA

        self.env = env

        #tensorflow :D
        self.sess = tf.Session()
        # #TO/DO:REMOVE THIS BIT
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        # info of the environment to pass to the agent
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        #FOR the RNN
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=300, state_is_tuple=True, reuse=None)
        cell_target = tf.contrib.rnn.BasicLSTMCell(num_units=300, state_is_tuple=True, reuse=None)
        self.ruido = OUNoise(self.action_dim, mu=0.4)  # this is the Ornstein-Uhlenbeck Noise
        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim, np.float64(1) , ACTOR_LEARNING_RATE, TAU)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim, CRITIC_LEARNING_RATE, TAU, cell, cell_target,
                               self.actor.get_num_trainable_vars())

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize replay memory
        if replay_buffer == None:
            self.replay_buffer = ReplayBuffer(self, BUFFER_SIZE)
        else:
            self.replay_buffer = replay_buffer

    def get_state_value(self, state):
        # q_values, _ = self.get_q_values(state)
        # q_value = max(q_values)
        # return q_value
        return self.critic.predict_target(np.array([state.flatten()]),
                                          self.scale(self.actor.predict(np.reshape(state,(1, self.state_dim))))
                                          ,20)[0][0]

    def get_action(self, state):
        """Returns action for obs

        Return policy based on exploration strategy of the TDLearning object.

        When an exploration method is added make sure the method is added in
        the class attributes and below for ease of usage.

        Parameters
        ----------
        obs : :obj:`list` of :obj:`int` or `np.ndarray`
            observation

        Returns
        -------
        :obj:`int`
            next action

        Raises
        ------
        NotImplementedError
            Please choose from the available exploration methods, see class
            attributes.

        """
        # 1. get action with actor, and add noise
        action_original = self.actor.predict(np.reshape(state,(1, self.state_dim)))  # + (10. / (10. + i))* np.random.randn(1)
        action = action_original + max(self.epsilon, 0) * self.ruido.noise()
        return self.scale(action)

    def scale(self, actions):  # assume domain [-1,1]
        actions = np.clip(actions, -1, 1)
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_actions = (((actions + 1) / 2) * (high - low)) + low
        return scaled_actions  # edit - DH.

    def observe(self, state, action, reward, new_state, done):
        if self.train_indicator:
            # 3. Save in replay buffer:
            self.replay_buffer.add(self, np.reshape(state, (self.actor.s_dim,)), np.reshape(action, (self.actor.a_dim,)), reward,
                              done, np.reshape(new_state, (self.actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if self.replay_buffer.size() > self.MINIBATCH_SIZE:

                # 4. sample random minibatch of transitions:
                s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.MINIBATCH_SIZE)

                # Calculate targets

                # 5. Train critic Network (states,actions, R + gamma* V(s', a')):
                # 5.1 Get critic prediction = V(s', a')
                # the a' is obtained using the actor prediction! or in other words : a' = actor(s')
                actor_predicted = self.actor.predict_target(s2_batch)
                target_q = self.critic.predict_target(s2_batch, actor_predicted, 20)

                # 5.2 get y_t where:
                y_i = []
                for k in range(self.MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + self.GAMMA * target_q[k])

                # 5.3 Train Critic!
                predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (self.MINIBATCH_SIZE, 1)), 20)

                # 6 Compute Critic gradient (depends on states and actions)
                # 6.1 therefore I first need to calculate the actions the current actor would take.
                a_outs = self.actor.predict(s_batch)

                # 6.2 I calculate the gradients
                grads = self.critic.action_gradients(s_batch, a_outs, 20)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

    def start_new_episode(self, state):
        self.epsilon -= (self.epsilon / self.EXPLORE)
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def render(self, env, **kwargs):
        return env.render()

    def end_episode(self):
        self.ruido.reset()

if __name__ == "__main__":
    import random
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain
    from smartstart.utilities.utilities import set_global_seeds
    from smartstart.utilities.plot import plot_summary, show_plot, \
        mean_reward_episode, steps_episode

    # Reset the seed for random number generation
    RANDOM_SEED = 1234

    set_global_seeds(RANDOM_SEED)

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    # Initialize agent, see class for available parameters
    agent = DDPG_agent(env, MINIBATCH_SIZE=40)

    # Train the agent, summary contains training data
    summary = rlTrain(agent, env, render=True,
                      render_episode=False,
                      print_results=True, num_episodes=1000)  # type: Summary

    summary.save(get_default_directory("ddpg_summaries"), extra_name_append="-1000ep")
