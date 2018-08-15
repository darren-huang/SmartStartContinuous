import argparse
import time
import tensorflow as tf

from RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from environments.continuous_mountain_car_editted import Continuous_MountainCarEnv_Editted
from smartexploration.smartexplorationcontinuous import SmartStartContinuous
from utilities.utilities import set_global_seeds, get_default_data_directory

parser = argparse.ArgumentParser(description='Set some flages')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--noGpu', action='store_true', help='If included, stops the gpu from being used')
parser.add_argument('-s', '--seed', dest="seed", default=int(time.time() * 10) % (2 ** 32 - 1), type=int)
args = parser.parse_args()
noGpu = args.noGpu
RANDOM_SEED = args.seed

if __name__ == "__main__":
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain, rlTrainGraphSS

    episodes = 1000
    lastLayerTanh = True

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

    with tf.Graph().as_default() as graph:
        with tf.Session(config=tfConfig, graph=graph) as sess:
            # Reset the seed for random number generation
            set_global_seeds(RANDOM_SEED)
            env.seed(RANDOM_SEED)

            # Initialize base agent, see class for available parameters
            base_agent = DDPG_Baselines_agent(env, sess,
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
                                             lastLayerTanh=lastLayerTanh,
                                             finalizeGraph=False)

            #initialize the smartStartContinuous (see class for parameter specifics)
            smart_start_agent = SmartStartContinuous(base_agent, env, sess,

                                                     buffer_size=100000,
                                                     exploitation_param=1.,
                                                     exploration_param=2.,
                                                     eta=0.5,
                                                     eta_decay_factor=.99,
                                                     n_ss=2000,
                                                     print_ss_stuff=True,

                                                     nnd_mb_final_steps=10,
                                                     nnd_mb_steps_per_waypoint=1,
                                                     nnd_mb_mean_per_stepsize=1,
                                                     nnd_mb_std_per_stepsize=1,
                                                     nnd_mb_stepsizes_in_waypoint_radii=1,

                                                     nnd_mb_gamma=.75,
                                                     nnd_mb_horizontal_penalty_factor=.5,
                                                     nnd_mb_horizon=4,
                                                     nnd_mb_num_control_samples=500,
                                                     nnd_mb_path_shortcutting=True,
                                                     nnd_mb_steps_before_giving_up_on_waypoint=5,

                                                     nnd_mb_load_dir_name="default",
                                                     nnd_mb_load_existing_training_data=True,

                                                     nnd_mb_num_fc_layers=1,
                                                     nnd_mb_depth_fc_layers=32,
                                                     nnd_mb_batchsize=512,
                                                     nnd_mb_lr=0.001,
                                                     nnd_mb_nEpoch=30,
                                                     nnd_mb_fraction_use_new=0.9,
                                                     nnd_mb_num_episodes_for_aggregation=4,
                                                     nnd_mb_make_aggregated_dataset_noisy=True,
                                                     nnd_mb_make_training_dataset_noisy=True,
                                                     nnd_mb_noise_actions_during_MPC_rollouts=True,

                                                     nnd_mb_verbose=False)
            sess.graph.finalize()

            # Train the agent, summary contains training data
            # summary = rlTrain(smart_start_agent, env, render=args.render,
            #                   render_episode=False,
            #                   print_steps=False,
            #                   print_results=False,
            #                   num_episodes=episodes,
            #                   print_time=True)  # type: Summary
            summary = rlTrainGraphSS(smart_start_agent, env,
                                     render=args.render,
                                     render_episode=False,
                                     print_steps=False,
                                     print_results=False,
                                     num_episodes=episodes,
                                     plot_ss_stuff=True,
                                     print_time=True)

            #naming stuff
            noGpu_str = "-NoGPU" if noGpu else ""
            llTanh_str = "-LLTanh" if lastLayerTanh else ""

            #adds extras parameters to the summary dictionary
            summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)

            #save the summary
            fp = summary.save(get_default_data_directory("smart_start_continuous_summaries/0/"),
                              extra_name_append="-" + str(
                                  episodes) + "ep" + noGpu_str + "-noNorm" + llTanh_str + "-decayingNoise" + "-2000n_ss")

            # save the tensorflow graph
            train_writer = tf.summary.FileWriter(fp[:-5])
            train_writer.add_graph(sess.graph)
