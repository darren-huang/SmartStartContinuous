import gym

from smartstart.RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from smartstart.reinforcementLearningCore.rlTrain import rlTrain
from smartstart.utilities.experimenter import run_experiment
from smartstart.utilities.utilities import set_global_seeds, get_default_data_directory


def task_run_ddpg_baselines_mc(params):
    import tensorflow as tf

    print("\n\nprocess " + str(params['id']) + " has started" + "-"*200 + "\n")

    noGpu = True
    render = False
    replay_buffer = None


    RANDOM_SEED = params['RANDOM_SEED']
    episodes = params['episodes']
    decayingNoise = params['decayingNoise']
    sub_dir_name = params['sub_dir_name']

    buffer_size = params['buffer_size']
    batch_size = params['batch_size']
    num_train_iterations = params['num_train_iterations']
    num_steps_before_train = params['num_steps_before_train']
    ou_epsilon = params['ou_epsilon']
    ou_min_epsilon = params['ou_min_epsilon']
    ou_epsilon_decay_factor = params['ou_epsilon_decay_factor']
    ou_mu = params['ou_mu']
    ou_sigma = params['ou_sigma']
    ou_theta = params['ou_theta']
    actor_lr = params['actor_lr']
    actor_h1 = params['actor_h1']
    actor_h2 = params['actor_h1']/2
    critic_lr = params['critic_lr']
    critic_h1 = params['critic_h1']
    critic_h2 = params['critic_h1']/2
    gamma = params['gamma']
    tau = params['tau']
    layer_norm = params['layer_norm']
    normalize_observations = params['normalize_observations']
    normalize_returns = params['normalize_returns']
    critic_l2_reg = params['critic_l2_reg']
    enable_popart = params['enable_popart']
    clip_norm = params['clip_norm']
    reward_scale = params['reward_scale']
    lastLayerTanh = params['lastLayerTanh']

    get_extra_name = params['get_extra_name']

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    if noGpu:
        tfConfig = tf.ConfigProto(device_count={'GPU': 0})
    else:
        tfConfig = None

    file_location = None

    with tf.Graph().as_default() as graph:
        with tf.Session(config=tfConfig, graph=graph) as sess:
            # with tf.Session() as sess:
            # Reset the seed for random number generation
            set_global_seeds(RANDOM_SEED)
            env.seed(RANDOM_SEED)

            # Initialize agent, see class for available parameters
            agent = DDPG_Baselines_agent(env, sess,
                                         replay_buffer=replay_buffer,
                                         buffer_size=buffer_size,
                                         batch_size=batch_size,
                                         num_train_iterations=num_train_iterations,
                                         num_steps_before_train=num_steps_before_train,
                                         ou_epsilon=ou_epsilon,
                                         ou_min_epsilon=ou_min_epsilon,
                                         ou_epsilon_decay_factor=ou_epsilon_decay_factor,
                                         ou_mu=ou_mu,
                                         ou_sigma=ou_sigma,
                                         ou_theta=ou_theta,
                                         actor_lr=actor_lr,
                                         actor_h1=actor_h1,
                                         actor_h2=actor_h2,
                                         critic_lr=critic_lr,
                                         critic_h1=critic_h1,
                                         critic_h2=critic_h2,
                                         gamma=gamma,
                                         tau=tau,
                                         layer_norm=layer_norm,
                                         normalize_observations=normalize_observations,
                                         normalize_returns=normalize_returns,
                                         critic_l2_reg=critic_l2_reg,
                                         enable_popart=enable_popart,
                                         clip_norm=clip_norm,
                                         reward_scale=reward_scale,
                                         lastLayerTanh=lastLayerTanh
                                         )

            # Train the agent, summary contains training data
            summary = rlTrain(agent, env, render=render,
                              render_episode=False,
                              print_steps=False,
                              print_results=False, num_episodes=episodes,
                              progress_bar=True,
                              id=params['id'],
                              num_ticks=params['num_ticks'])  # type: Summary

            noGpu_str = "_NoGPU" if noGpu else ""
            llTanh_str = "_LLTanh" if lastLayerTanh else ""
            decayingNoise_str = "-decayingNoise" if decayingNoise else ""
            summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)
            fp = summary.save(get_default_data_directory("ddpg_baselines_summaries/" + sub_dir_name), last_name_section=True,
                              extra_name_append= get_extra_name(params) +
                              "_" + str(episodes) + "ep" + noGpu_str + "_noNorm" + llTanh_str + decayingNoise_str)

            print("\n\nprocess " + str(params['id']) + " has finished" + "!" * 200 + "\n")

            # train_writer = tf.summary.FileWriter(fp[:-35])
            # train_writer.add_graph(sess.graph)

# Define the task function for the experiment
def task_print(params):
    # print(sorted(params.items(), key=lambda x: x[0]))
    print(params['get_extra_name'](params))

def get_extra_name(params):
    critic_string = '_c-' + str(params['critic_h1']) + "-" + str(params['critic_h1']/2) + "-" + str(params['critic_lr']).replace(".","d")
    actor_string = '_a-' + str(params['actor_h1']) + "-" + str(params['actor_h1']/2) + "-" + str(params['actor_lr']).replace(".","d")
    return actor_string + critic_string

if __name__ == "__main__":
    experiment_task = task_run_ddpg_baselines_mc
    # experiment_task = task_print
    num_exp_per_param = 1
    episodes = 1000
    num_ticks = 200
    decaying_noise = True #must be the case that these match the parameters
    sub_dir_name = 'hidden_layer_size_experiment'

    paramsGrid = {
        'task' : experiment_task,
        'num_exp' : num_exp_per_param,

        'num_ticks' : [num_ticks],
        'RANDOM_SEED' : [1, 12, 123, 1234, 12345], #new seed so each experiment is different but reproducible
        'episodes' : [episodes],
        'decayingNoise' : [decaying_noise],
        'sub_dir_name' : [sub_dir_name],

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
        'actor_lr': [0.001, 0.005],
        'actor_h1': [64, 128,200], #h2 will be half of this
        'critic_lr': [0.001, 0.005],
        'critic_h1': [64, 128,200], #h2 will be half of this
        'gamma': [0.99],
        'tau': [0.001],
        'layer_norm': [False],
        'normalize_observations': [False],
        'normalize_returns': [False],
        'critic_l2_reg': [0],
        'enable_popart': [False],
        'clip_norm': [None],
        'reward_scale': [1.],
        'lastLayerTanh': [True],

        'get_extra_name' : [get_extra_name]


    }

    run_experiment(paramsGrid, n_processes=-1)
