import gym
import random

from smartstart.RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from smartstart.smartexploration.smartexplorationcontinuous import SmartStartContinuous
from smartstart.reinforcementLearningCore.rlTrain import rlTrain
from smartstart.utilities.experimenter import run_experiment, create_experimeter_info_txt
from smartstart.utilities.utilities import set_global_seeds, get_default_data_directory


def task_run_ss_ddpg_baselines_mc(params):
    import tensorflow as tf

    print("\n\nprocess " + str(params['id']) + " has started" + "-"*200 + "\n")

    noGpu = params['noGpu']
    render = False
    replay_buffer = None

    # random seed each time
    random.seed()
    RANDOM_SEED = random.randint(0, 2**32 - 1)

    # Overall Options
    episodes = params['episodes']
    dir_name = params['dir_name']

    #naming function
    get_extra_name = params['get_extra_name']

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    if noGpu:
        tfConfig = tf.ConfigProto(device_count={'GPU': 0})
    else:
        tfConfig = None

    with tf.Graph().as_default() as graph:
        with tf.Session(config=tfConfig, graph=graph) as sess:
            # with tf.Session() as sess:
            # Reset the seed for random number generation
            set_global_seeds(RANDOM_SEED)
            env.seed(RANDOM_SEED)

            # Initialize agent, see class for available parameters
            base_agent = DDPG_Baselines_agent(env, sess,
                                         replay_buffer=replay_buffer,
                                         buffer_size=params['buffer_size'],
                                         batch_size = params['batch_size'],
                                         num_train_iterations = params['num_train_iterations'],
                                         num_steps_before_train = params['num_steps_before_train'],
                                         ou_epsilon = params['ou_epsilon'],
                                         ou_min_epsilon = params['ou_min_epsilon'],
                                         ou_epsilon_decay_factor = params['ou_epsilon_decay_factor'],
                                         ou_mu = params['ou_mu'],
                                         ou_sigma = params['ou_sigma'],
                                         ou_theta = params['ou_theta'],
                                         actor_lr = params['actor_lr'],
                                         actor_h1 = params['actor_h1'],
                                         actor_h2 = params['actor_h2'],
                                         critic_lr = params['critic_lr'],
                                         critic_h1 = params['critic_h1'],
                                         critic_h2 = params['critic_h2'],
                                         gamma = params['gamma'],
                                         tau = params['tau'],
                                         layer_norm = params['layer_norm'],
                                         normalize_observations = params['normalize_observations'],
                                         normalize_returns = params['normalize_returns'],
                                         critic_l2_reg = params['critic_l2_reg'],
                                         enable_popart = params['enable_popart'],
                                         clip_norm = params['clip_norm'],
                                         reward_scale = params['reward_scale'],
                                         lastLayerTanh = params['lastLayerTanh'],
                                         finalizeGraph=False
                                         )

            smart_start_agent = SmartStartContinuous(base_agent, env, sess,
                                                    buffer_size=params['buffer_size'],
                                                    exploitation_param = params['exploitation_param'],
                                                    exploration_param = params['exploration_param'],
                                                    eta = params['eta'],
                                                    eta_decay_factor = params['eta_decay_factor'],
                                                    n_ss = params['n_ss'],
                                                    print_ss_stuff=False,
                                                    # sigma=params['sigma'],
                                                    # smart_start_selection_modified_distance_function=params['smart_start_selection_modified_distance_function'],
                                                    nnd_mb_final_steps = params['nnd_mb_final_steps'],
                                                    nnd_mb_steps_per_waypoint = params['nnd_mb_steps_per_waypoint'],
                                                    nnd_mb_mean_per_stepsize = params['nnd_mb_mean_per_stepsize'],
                                                    nnd_mb_std_per_stepsize = params['nnd_mb_std_per_stepsize'],
                                                    nnd_mb_stepsizes_in_waypoint_radii = params['nnd_mb_stepsizes_in_waypoint_radii'],
                                                    nnd_mb_gamma = params['nnd_mb_gamma'],
                                                    nnd_mb_horizontal_penalty_factor = params['nnd_mb_horizontal_penalty_factor'],
                                                    nnd_mb_horizon = params['nnd_mb_horizon'],
                                                    nnd_mb_num_control_samples = params['nnd_mb_num_control_samples'],
                                                    nnd_mb_load_dir_name = params['nnd_mb_load_dir_name'],
                                                    nnd_mb_load_existing_training_data = params['nnd_mb_load_existing_training_data'],
                                                    nnd_mb_num_fc_layers = params['nnd_mb_num_fc_layers'],
                                                    nnd_mb_depth_fc_layers = params['nnd_mb_depth_fc_layers'],
                                                    nnd_mb_batchsize = params['nnd_mb_batchsize'],
                                                    nnd_mb_lr = params['nnd_mb_lr'],
                                                    nnd_mb_nEpoch = params['nnd_mb_nEpoch'],
                                                    nnd_mb_fraction_use_new = params['nnd_mb_fraction_use_new'],
                                                    nnd_mb_num_episodes_for_aggregation = params['nnd_mb_num_episodes_for_aggregation'],
                                                    nnd_mb_make_aggregated_dataset_noisy = params['nnd_mb_make_aggregated_dataset_noisy'],
                                                    nnd_mb_make_training_dataset_noisy = params['nnd_mb_make_training_dataset_noisy'],
                                                    nnd_mb_noise_actions_during_MPC_rollouts = params['nnd_mb_noise_actions_during_MPC_rollouts'],
                                                    nnd_mb_verbose = params['nnd_mb_verbose'])

            sess.graph.finalize()

            # Train the agent, summary contains training data
            summary = rlTrain(smart_start_agent, env, render=render,
                              render_episode=False,
                              print_steps=False,
                              print_results=False, num_episodes=episodes,
                              progress_bar=True,
                              id=params['id'],
                              num_ticks=params['num_ticks'])  # type: Summary

            summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)
            fp = summary.save(get_default_data_directory(dir_name), last_name_section=True,
                              extra_name_append= get_extra_name(params))

            print("\n\nprocess " + str(params['id']) + " has finished" + "!" * 200 + "\n")

            # train_writer = tf.summary.FileWriter(fp[:-35])
            # train_writer.add_graph(sess.graph)

# Define the task function for the experiment
def task_print(params):
    # print(sorted(params.items(), key=lambda x: x[0]))
    print(params['get_extra_name'](params))

def get_extra_name(params):
    exploration_param_str = "_explorP-" + str(params['exploration_param'])
    eta_decay_str = "_etaDecay-" + str(params['eta_decay_factor'])
    return exploration_param_str + eta_decay_str

if __name__ == "__main__":
    experiment_task = task_run_ss_ddpg_baselines_mc

    #changeable parameter
    num_exp_per_param = 25
    episodes = 1000
    noGpu = True
    lastLayerTanh = True


    #naming / display
    num_ticks = 200 #ticks to display while the process is running
    decaying_noise = True #must be the case that these match the parameters
    dir_name = 'smart_start_continuous_summaries/ddpg_baselines/hyper_parameter_search'

    paramsGrid = {
        'task' : experiment_task,
        'num_exp' : num_exp_per_param,

        'num_ticks' : [num_ticks],
        'episodes' : [episodes],
        'decayingNoise' : [decaying_noise],
        'dir_name' : [dir_name],
        'noGpu' : [noGpu],

        'buffer_size': [100000],
        'batch_size': [64],
        'num_train_iterations': [1],
        'num_steps_before_train': [1],
        'ou_epsilon': [1.0],
        'ou_min_epsilon': [0.01],
        'ou_epsilon_decay_factor': [.99],
        'ou_mu': [0.4],
        'ou_sigma': [0.6],
        'ou_theta': [.15],
        'actor_lr': [0.001], # these are the optimal critic and actor sizes : )
        'actor_h1': [64],
        'actor_h2': [32],
        'critic_lr': [0.001],
        'critic_h1': [64],
        'critic_h2': [32],
        'gamma': [0.99],
        'tau': [0.001],
        'layer_norm': [False],
        'normalize_observations': [False],
        'normalize_returns': [False],
        'critic_l2_reg': [0],
        'enable_popart': [False],
        'clip_norm': [None],
        'reward_scale': [1.],
        'lastLayerTanh': [lastLayerTanh],

        'exploitation_param': [1.],
        'exploration_param': [1., 2., 5.], #NOT CONSTANT
        'eta': [0.5],
        'eta_decay_factor': [.99, .98], # NOT CONSTANT
        'n_ss': [2000],
        # 'sigma': [1],
        # 'smart_start_selection_modified_distance_function': [True, False], #NOT CONSTANT
        'nnd_mb_final_steps': [10],
        'nnd_mb_steps_per_waypoint': [1],
        'nnd_mb_mean_per_stepsize': [1],
        'nnd_mb_std_per_stepsize': [1],
        'nnd_mb_stepsizes_in_waypoint_radii': [1],
        'nnd_mb_gamma': [.75],
        'nnd_mb_horizontal_penalty_factor': [.5],
        'nnd_mb_horizon': [4],
        'nnd_mb_num_control_samples': [5000],
        'nnd_mb_load_dir_name': ["default"],
        'nnd_mb_load_existing_training_data': [True],
        'nnd_mb_num_fc_layers': [1],
        'nnd_mb_depth_fc_layers': [500],
        'nnd_mb_batchsize': [512],
        'nnd_mb_lr': [0.001],
        'nnd_mb_nEpoch': [30],
        'nnd_mb_fraction_use_new': [0.9],
        'nnd_mb_num_episodes_for_aggregation': [4],
        'nnd_mb_make_aggregated_dataset_noisy': [True],
        'nnd_mb_make_training_dataset_noisy': [True],
        'nnd_mb_noise_actions_during_MPC_rollouts': [True],
        'nnd_mb_verbose': [False],

        'get_extra_name' : [get_extra_name]
    }

    noGpu_str = "_NoGPU" if noGpu else ""
    llTanh_str = "_LLTanh" if lastLayerTanh else ""
    decayingNoise_str = "_decayingNoise" if decaying_noise else ""
    create_experimeter_info_txt(paramsGrid, get_default_data_directory(dir_name),
                                name_append= "_" + str(episodes) + "ep" + noGpu_str + llTanh_str + decayingNoise_str)
    run_experiment(paramsGrid, n_processes=-1)
