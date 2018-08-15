import argparse

import tensorflow as tf

from RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from environments.continuous_mountain_car_editted import Continuous_MountainCarEnv_Editted
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.utilities import get_default_data_directory, set_global_seeds



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
    # ENV_NAME = 'MountainCarContinuous-v0'
    # env = gym.make(ENV_NAME)
    env = Continuous_MountainCarEnv_Editted.make_timed_env(.33,
                                                           max_episode_steps=1000,
                                                           max_episode_seconds=None)

    if noGpu:
        tfConfig = tf.ConfigProto(device_count={'GPU': 0})
    else:
        tfConfig = None

    with tf.Session(config=tfConfig) as sess:
        # Reset the seed for random number generation
        set_global_seeds(RANDOM_SEED)
        env.seed(RANDOM_SEED)

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
                                     actor_h1=64,
                                     actor_h2=32,
                                     critic_lr=0.001,
                                     critic_h1=64,
                                     critic_h2=32,
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
        summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)
        fp = summary.save(get_default_data_directory("ddpg_baselines_summaries/1"),
                          extra_name_append="-" + str(episodes) + "ep" + noGpu_str +"-noNorm" + llTanh_str + "-decayingNoise")


        # train_writer = tf.summary.FileWriter(fp[:-5])
        # train_writer.add_graph(sess.graph)
