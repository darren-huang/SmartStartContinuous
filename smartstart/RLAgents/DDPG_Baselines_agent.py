from abc import ABCMeta

from smartstart.reinforcementLearningCore.agents import ValueFuncRLAgent, ReplayBufferRLAgent
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.utilities.utilities import get_default_directory, set_global_seeds
from smartstart.utilities.datacontainers import Summary
import tensorflow as tf

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import *

import numpy as np

class ReplayBufferWrapper: #wraps a ReplayBuffer but has the same interface as baselines' ddpg memeory
    """
    specifically for the DDPG_Baseline to use
    """
    def __init__(self, replay_buffer, main_agent):
        """
        :param replay_buffer: replay buffer to be wrapped, if replay buffer is None, a new one is created
        :param buffer_size: the size of the buffer (only set IF replayBuffer=None)
        :param random_seed: the random seed of the buffer (only set IF replayBuffer=None)
        """
        self.replay_buffer = replay_buffer #type: ReplayBuffer
        self.main_agent = main_agent

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)

        result = {
            'obs0': s_batch,
            'obs1': s2_batch,
            'rewards': r_batch.reshape((-1,1)),
            'actions': a_batch,
            'terminals1': t_batch.reshape((-1,1)),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.replay_buffer.add(self.main_agent, obs0, action, reward, terminal1, obs1)

    @property
    def nb_entries(self):
        return len(self.replay_buffer)


class DDPG_Baselines_agent(ValueFuncRLAgent, ReplayBufferRLAgent):
    #TODO edit this description
    """
    Continuous RLAgent using the DDPG from the baselines library using OrnsteinUhlenbeckActionNoise
    """

    def __init__(self, env, sess, replay_buffer=None, layer_norm=True, mu=0.4, sigma=0.2, BUFFER_SIZE=10000, gamma=0.99,
                 tau=0.001, normalize_returns=False, normalize_observations=True, batch_size=64, actor_lr=1e-4,
                 critic_lr=1e-3, critic_l2_reg=0, enable_popart=False, clip_norm=None, reward_scale=1.,
                 num_steps_before_train=40, num_train_iterations=50):

        super().__init__()
        self.env = env
        self.num_steps_before_train = num_steps_before_train
        self.remaining_steps_before_train = num_steps_before_train
        self.num_train_iterations = num_train_iterations
        self.batch_size = batch_size

        #tensorflow :D
        self.sess = sess
        # #TO/DO:REMOVE THIS BIT
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess) # debug tensorflow


        # info of the environment to pass to the agent
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # action_noise = None
        param_noise = None
        nb_actions = env.action_space.shape[-1]

        #TODO: set an option to choose which noise
        # param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev)) #diff types of noise
        # action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions)) # diff types of noise
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(nb_actions) * float(mu),
                                                            sigma=float(sigma) * np.ones(nb_actions))

        if replay_buffer:
            self.replay_buffer = replay_buffer  # type: ReplayBuffer
        else:
            self.replay_buffer = ReplayBuffer(self, BUFFER_SIZE)
        self.memory = ReplayBufferWrapper(self.replay_buffer, self)

        critic = Critic(layer_norm=layer_norm)
        actor = Actor(nb_actions, layer_norm=layer_norm)

        self.inner_ddpg_agent = DDPG(actor, critic, self.memory, env.observation_space.shape, env.action_space.shape,
                                     gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                                     normalize_observations=normalize_observations,
                                     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                                     critic_l2_reg=critic_l2_reg,
                                     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=enable_popart, clip_norm=clip_norm,
                                     reward_scale=reward_scale)

        self.inner_ddpg_agent.initialize(self.sess)
        self.sess.graph.finalize()
        self.inner_ddpg_agent.reset()



    def get_state_value(self, state):
        raw_action, q = self.inner_ddpg_agent.pi(state, apply_noise=True,
                                                 compute_Q=True)  # DDPG action is always [-1,1]
        return q

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
        #TODO: see if we can add an epsilon to control how much noise is added (reduce noise overtime)
        raw_action, q = self.inner_ddpg_agent.pi(state, apply_noise=True, compute_Q=True) # DDPG action is always [-1,1]
        return self.scale(self.scale(raw_action))

    def scale(self, actions):  # assume domain [-1,1]
        actions = np.clip(actions, -1, 1)
        low, high = self.env.action_space.low, self.env.action_space.high
        scaled_actions = (((actions + 1) / 2) * (high - low)) + low
        return scaled_actions  # edit - DH.

    def observe(self, state, action, reward, new_state, done):
        self.inner_ddpg_agent.store_transition(state, action, reward, new_state, done)
        self.remaining_steps_before_train -= 1

        if self.remaining_steps_before_train <= 0:
            self.train()




    def start_new_episode(self, state):
        pass

    def render(self, env, **kwargs):
        return env.render()

    def end_episode(self):
        self.inner_ddpg_agent.reset()
        self.train()

    #private method
    def train(self):
        if len(self.memory.replay_buffer) >= self.batch_size:
            self.remaining_steps_before_train = self.num_steps_before_train
            cl, al = None, None
            for train_iter in range(self.num_train_iterations):
                # following segment is for the parameter noise (currently only using OrnsteinUhlenbeckActionNoise)
                # if self.memory.nb_entries >= self.batch_size and t % param_noise_adaption_interval == 0:
                #     distance = agent.adapt_param_noise()
                cl, al = self.inner_ddpg_agent.train()
                self.inner_ddpg_agent.update_target_net()
            #TODO: have an option for printing this out
            print("Critic Loss: " + str(cl) + ", Actor Loss: " + str(al), end='')


if __name__ == "__main__":
    import random
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain
    from smartstart.utilities.plot import plot_summary, show_plot, \
        mean_reward_episode, steps_episode

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    # Reset the seed for random number generation
    RANDOM_SEED = 1234
    set_global_seeds(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    with tf.Session() as sess:
        # Initialize agent, see class for available parameters
        agent = DDPG_Baselines_agent(env, sess, BUFFER_SIZE=10000, actor_lr=0.01, critic_lr=0.005,
                                     num_steps_before_train=1, num_train_iterations=1)

        # Train the agent, summary contains training data
        summary = rlTrain(agent, env, render=True,
                          render_episode=False,
                          print_results=True, num_episodes=1000)  # type: Summary

    summary.save(get_default_directory("ddpg_baselines_summaries"), extra_name_append="-1000ep")
