import numpy as np
import argparse
import tensorflow as tf
from baselines.ddpg.ddpg import DDPG
# from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import *
from smartstart.RLContinuousAlgorithms.DDPG_Baselines_editted.models_editted import Actor_Editted, Critic_Editted

from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.reinforcementLearningCore.agents_abstract_classes import ValueFuncRLAgent, ReplayBufferRLAgent
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.utilities import get_default_data_directory, set_global_seeds


class ReplayBufferWrapper:  # wraps a ReplayBuffer but has the same interface as baselines' ddpg memeory
    """
    specifically for the DDPG_Baseline to use
    """

    def __init__(self, replay_buffer, main_agent):
        """
        :param replay_buffer: replay buffer to be wrapped, if replay buffer is None, a new one is created
        :param buffer_size: the size of the buffer (only set IF replayBuffer=None)
        :param random_seed: the random seed of the buffer (only set IF replayBuffer=None)
        """
        self.replay_buffer = replay_buffer  # type: ReplayBuffer
        self.main_agent = main_agent

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)

        result = {
            'obs0': s_batch,
            'obs1': s2_batch,
            'rewards': r_batch.reshape((-1, 1)),
            'actions': a_batch,
            'terminals1': t_batch.reshape((-1, 1)),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.replay_buffer.add(self.main_agent, obs0, action, reward, terminal1, obs1)

    @property
    def nb_entries(self):
        return len(self.replay_buffer)


class DecayingOrnsteinUhlenbeckActionNoise(OrnsteinUhlenbeckActionNoise):
    def __init__(self, epsilon, min_epsilon, epsilon_decay_factor, mu, sigma, theta=.15, dt=1e-2, x0=None):
        super().__init__(mu, sigma, theta=theta, dt=dt, x0=x0)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_factor = epsilon_decay_factor

    def __call__(self):
        x = super().__call__()
        scaled_x = x * max(self.epsilon, 0)
        return scaled_x

    def reduce_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay_factor, self.min_epsilon)


class DDPG_Baselines_agent(ValueFuncRLAgent, ReplayBufferRLAgent):
    # TODO edit this description
    """
    Continuous RLAgent using the DDPG from the baselines library using OrnsteinUhlenbeckActionNoise
    """

    def __init__(self, env, sess, replay_buffer=None, buffer_size=10000, batch_size=64, num_train_iterations=50,
                 num_steps_before_train=40,
                 ou_epsilon=1.0, ou_min_epsilon=0.01, ou_epsilon_decay_factor=.99,
                 ou_mu=0.4, ou_sigma=0.2, ou_theta=.15, actor_lr=1e-4, actor_h1=64, actor_h2=64, critic_lr=1e-3,
                 critic_h1=64, critic_h2=64, gamma=0.99, tau=0.001, layer_norm=False, normalize_observations=False,
                 normalize_returns=False, critic_l2_reg=0, enable_popart=False, clip_norm=None, reward_scale=1.,
                 lastLayerTanh=False):
        """

        :param env:
        :param sess:
        :param replay_buffer:
        :param buffer_size:  # size of the replay buffer
        :param layer_norm: adds a layer of normalization after fully connected layers (https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm)
                           Effects all layers in both Actor and Critic (4 layers total, 2 per network) and same for target networks i think
        :param ou_epsilon: Initial value of epsilon -> factor multiplied with the noise (MUST BE POSITIVE)
        :param ou_min_epsilon: over time (after each episode) epsilon keeps decreasing, this is the lowest amount it can decrease to (MUST BE POSITIVE)
        :param ou_epsilon_decay_factor: roughly the number of episodes b4 the epsilon reaches the minimum (each episode, reduces epsilon by (epsilon/ou_nb_explore_ep)
        :param ou_mu:  # OrnsteinUhlenbeckActionNoise MEAN https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
        :param ou_sigma: # OrnsteinUhlenbeckActionNoise degree of volatility
        :param ou_theta: # OrnsteinUhlenbeckActionNoise how much to pull back towards the mean
        :param gamma: # discount factor -> used for the Critic network training
        :param tau: # soft target updates (parameters of Target networks get updated by tau*parameters of real networks)
        :param normalize_returns: # Normalizes inputs into the Target networks I think (maybe only target Critic) then denormalizes afterwards
        :param normalize_observations: # Batch normalizations -> normalizes observations fed into Critic optimizer (actor network too)
        :param batch_size: #Training is done in batches, this determines how many samples from replay buffer is in each batch
        :param actor_lr:  # Base learning rate for the Actor Network
        :param critic_lr: # Base learning rate for the Critic Network
        :param critic_l2_reg: # Base learning rate for the Critic Regularization Network
        :param actor_h1: # size of hidden layer 1 for actor
        :param actor_h2: # size of hidden layer 2 for actor
        :param critic_h1: # size of hidden layer 1 for critic
        :param critic_h2: # size of hidden layer 2 for critic
        :param enable_popart: # (Preserving Outputs Precisely, while Adaptively Rescaling Targets) see https://arxiv.org/pdf/1602.07714.pdf
        :param clip_norm: # normalizes some gradient (reduces the size below some max) see https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
        :param reward_scale: # scales reward
        :param num_steps_before_train: Agent will keep taking this many steps, after which it will run self.train()
        :param num_train_iterations: When self.train() is called, it will run the train operation this many times (each train operations uses batch_size batches)
        """

        super().__init__()
        self.param_dict = locals().copy()
        for var_str in ['__class__', 'self', 'sess', 'env']:
            self.param_dict[var_str] = "Not serializable"

        self.env = env
        self.num_steps_before_train = num_steps_before_train
        self.remaining_steps_before_train = num_steps_before_train
        self.num_train_iterations = num_train_iterations
        self.batch_size = batch_size

        # tensorflow :D
        self.sess = sess
        # #TO/DO:REMOVE THIS BIT
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess) # debug tensorflow

        with self.sess.as_default():
            # info of the environment to pass to the agent
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]

            # action_noise = None
            param_noise = None
            nb_actions = env.action_space.shape[-1]

            # TODO: set an option to choose which noise
            # param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev)) #diff types of noise
            # action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions)) # diff types of noise
            # action_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(nb_actions) * float(ou_mu),
            #                                             theta=,
            #                                             sigma=float(ou_sigma) * np.ones(nb_actions))
            self.decaying_ou_action_noise = DecayingOrnsteinUhlenbeckActionNoise(epsilon=ou_epsilon,
                                                                min_epsilon=ou_min_epsilon,
                                                                epsilon_decay_factor=ou_epsilon_decay_factor,
                                                                mu=np.ones(nb_actions) * float(ou_mu),
                                                                theta=ou_theta,
                                                                sigma=float(ou_sigma) * np.ones(nb_actions))

            if replay_buffer:
                self.replay_buffer = replay_buffer  # type: ReplayBuffer
            else:
                self.replay_buffer = ReplayBuffer(self, buffer_size)
            self.memory = ReplayBufferWrapper(self.replay_buffer, self)

            critic = Critic_Editted(layer_norm=layer_norm, h1=critic_h1, h2=critic_h2, lastLayerTanh=lastLayerTanh)
            actor = Actor_Editted(nb_actions, layer_norm=layer_norm, h1=actor_h1, h2=actor_h2, lastLayerTanh=lastLayerTanh)

            self.inner_ddpg_agent = DDPG(actor, critic, self.memory, env.observation_space.shape,
                                         env.action_space.shape,
                                         gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                                         normalize_observations=normalize_observations,
                                         batch_size=batch_size, action_noise=self.decaying_ou_action_noise,
                                         param_noise=param_noise,
                                         critic_l2_reg=critic_l2_reg,
                                         actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=enable_popart,
                                         clip_norm=clip_norm,
                                         reward_scale=reward_scale)

            self.inner_ddpg_agent.initialize(self.sess)
            # TODO comment/uncomment
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
        # TODO: see if we can add an epsilon to control how much noise is added (reduce noise overtime)
        raw_action, q = self.inner_ddpg_agent.pi(state, apply_noise=True,
                                                 compute_Q=True)  # DDPG action is always [-1,1]
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
        self.decaying_ou_action_noise.reduce_epsilon()
        self.train()

    def get_param_dict(self):
        return self.param_dict

    # private method
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
            # TODO: have an option for printing this out
            print("Critic Loss: " + str(cl) + ", Actor Loss: " + str(al), end='')


import time

parser = argparse.ArgumentParser(description='Set some flages')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--noGpu', action='store_true', help='If included, stops the gpu from being used')
parser.add_argument('-s', '--seed', dest="seed", default=int(time.time() * 10) % (2 ** 32 - 1), type=int)
args = parser.parse_args()
noGpu = args.noGpu
RANDOM_SEED = args.seed

if __name__ == "__main__":
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    if noGpu:
        tfConfig = tf.ConfigProto(device_count={'GPU': 0})
    else:
        tfConfig = None

    file_location = None
    with tf.Session(config=tfConfig) as sess:
        # with tf.Session() as sess:
        # Reset the seed for random number generation
        set_global_seeds(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        a = tf.random_uniform([1])
        print(sess.run(a))  # generates 'A1'
        print(sess.run(a))  # generates 'A2'
        print(sess.run(a))  # generates 'B1'
        print(sess.run(a))  # generates 'B2'

        # run parameters
        episodes = 1000
        lastLayerTanh = True

        # Initialize agent, see class for available parameters
        agent = DDPG_Baselines_agent(env, sess,
                                     replay_buffer=None,
                                     buffer_size=100000,
                                     batch_size=64,
                                     num_train_iterations=1,
                                     num_steps_before_train=1,
                                     ou_epsilon=1.0,
                                     ou_min_epsilon=0.01,
                                     ou_epsilon_decay_factor=.99,
                                     ou_mu=0.4,
                                     ou_sigma=0.6,
                                     ou_theta=.15,
                                     actor_lr=0.001,
                                     actor_h1=200,
                                     actor_h2=100,
                                     critic_lr=0.005,
                                     critic_h1=200,
                                     critic_h2=100,
                                     gamma=0.99,
                                     tau=0.001,
                                     layer_norm=False,
                                     normalize_observations=False,
                                     normalize_returns=False,
                                     critic_l2_reg=0,
                                     enable_popart=False,
                                     clip_norm=None,
                                     reward_scale=1.,
                                     lastLayerTanh=lastLayerTanh
                                     )

        # Train the agent, summary contains training data
        summary = rlTrain(agent, env, render=args.render,
                          render_episode=False,
                          print_results=True, num_episodes=episodes)  # type: Summary

        noGpu_str = "-NoGPU" if noGpu else ""
        llTanh_str = "-LLTanh" if lastLayerTanh else ""
        fp = summary.save(get_default_data_directory("ddpg_baselines_summaries_1"),
                          extra_name_append="-" + str(episodes) + "ep" + noGpu_str +"-noNorm" + llTanh_str + "-decayingNoise")
        summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)

        train_writer = tf.summary.FileWriter(fp[:-5])
        train_writer.add_graph(sess.graph)
