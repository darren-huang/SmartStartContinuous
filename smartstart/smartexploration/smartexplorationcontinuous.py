"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object.
"""
import argparse
import scipy.stats
import time

import numpy as np
import tensorflow as tf

from smartstart.RLAgents.DDPG_Baselines_agent import DDPG_Baselines_agent
from smartstart.RLAgents.NND_MB_agent import NND_MB_agent
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.reinforcementLearningCore.agents_abstract_classes import RLAgent, ValueFuncRLAgent, ReplayBufferRLAgent
from smartstart.utilities.datacontainers import Summary
from smartstart.utilities.utilities import get_default_data_directory, set_global_seeds
from utilities.numerical import volume_of_n_dimensional_hyperellipsoid
from utilities.plot import plot_2d_density


class SmartStartContinuous(RLAgent):
    """SmartStart

    SmartStart algorithm consists of two stages:

        1.  Select smart start
        2.  Guide to smart start


    1. The smart start is selected using the UCB1 algorithm,
        as described at by the get_start method. UCB1 algorithm is for the multi-arm bandit problem
        (envision the problem as finding which slot machine arm to pull)
        Therefore the Smart Start Selection has 2 aspects to it.
            a. State value estimation: this is supposed to behave like the "reward" recieved from previous
                "pulls" of the slot machine. The base agent must be able to predict state values (expected reward at a given state)
            b. State visitation estimation: this is supposed to behave like the "number of times" a slot machine has been "pulled"
                This is estimated by using multivariate Kernel Density Estimation its implementation can be found at
            #TODO link SciPy KDE method

    2. The Smart Start Navigation is done using a Neural Network Dynamics estimator and a model predictive controller that
        generates random action sequences, predicts where the agent would go, and chooses the first action of the best
        sequence. #TODO link the neural network dynamics paper
        The Smart Start Navigation rather than trying to directly walk to the smart start state (which is often
        impossible to go towards in a straight line), it will try to walk the path it previously took to get to the smart
        start state. The "best action sequence" is the one that maximizes the built in reward function.
            The reward function rewards shortening the distance left on the path to get the goal and penalizes the agent
            for being too far perpendicularly from the current line segment it is following
        You can find the algorithm used for this inside 'smartstart/RLAgents/NND_MB_agent.py'

    After reaching the smart start the agent will continue with the
    normal reinforcement learning as described by the base class.
    """

    def __init__(self, agent, env, sess,
                 buffer_size=500000,
                 exploitation_param=1.,
                 exploration_param=2.,
                 eta=0.5,
                 eta_decay_factor=1.,
                 n_ss=1000,
                 print_ss_stuff=True,

                 nnd_mb_final_steps=10,
                 nnd_mb_steps_per_waypoint=1,
                 nnd_mb_mean_per_stepsize=1,
                 nnd_mb_std_per_stepsize=1,
                 nnd_mb_stepsizes_in_waypoint_radii=1,

                 nnd_mb_gamma=.75,
                 nnd_mb_horizontal_penalty_factor=.5,
                 nnd_mb_horizon=20,
                 nnd_mb_num_control_samples=5000,
                 nnd_mb_path_shortcutting=True,
                 nnd_mb_steps_before_giving_up_on_waypoint=5,

                 nnd_mb_save_dir_name="save_untitled",
                 nnd_mb_load_dir_name="untitled_load",
                 nnd_mb_save_training_data=False,
                 nnd_mb_save_resulting_dynamics_model=False,
                 nnd_mb_load_existing_training_data=False,
                 nnd_mb_load_existing_dynamics_model=False,

                 nnd_mb_num_fc_layers=1,
                 nnd_mb_depth_fc_layers=500,
                 nnd_mb_batchsize=512,
                 nnd_mb_lr=0.001,
                 nnd_mb_nEpoch=30,
                 nnd_mb_fraction_use_new=0.9,
                 nnd_mb_num_episodes_for_aggregation=3,
                 nnd_mb_make_aggregated_dataset_noisy=True,
                 nnd_mb_make_training_dataset_noisy=True,
                 nnd_mb_noise_actions_during_MPC_rollouts=True,

                 nnd_mb_verbose=True,

                 nnd_mb_use_threading=True,
                 nnd_mb_num_rollouts_train=25,
                 nnd_mb_num_rollouts_val=20,
                 nnd_mb_dt_steps=3,
                 nnd_mb_steps_per_rollout_train=333,
                 nnd_mb_steps_per_rollout_val=333):
        """

        :param agent: base agent smart start builds on top of
            NOTE: if the agent inherits from ReplayBufferRLAgent, then SmartStartContinuous will assign itself as the
                main observer of the replay buffer (but this allows them to share the same replay buffer)
        :param env: environment to run on
        :param sess: tensorflow session
        :param buffer_size: size of the buffer ONLY WORKS if base agent does NOT have a buffer already
        :param exploitation_param: used in the UCB algorithm/multi-arm bandit problem (choosing which smart start to use)
        :param exploration_param: used in the UCB algorithm/multi-arm bandit problem (choosing which smart start to use)
        :param eta: probability of smart start
        :param eta_decay_factor: the factor that multiplies with eta after each episode (example eta *= eta_decay_factor)
        :param n_ss: number of states from replay_buffer to consider as potential smartstart states
        :param print_ss_stuff: Printing out smart start messages

        for all parameters with 'nnd_mb' please see their descriptions under NND_MB_agent

        :type agent: ValueFuncRLAgent
        """
        self.param_dict = locals().copy()
        for var_str in ['__class__', 'self', 'sess', 'env']:
            self.param_dict[var_str] = "Not serializable"
        self.param_dict['agent'] = agent.get_param_dict()

        self.exploitation_param = exploitation_param
        self.exploration_param = exploration_param
        self.eta = eta
        self.eta_decay_factor = eta_decay_factor

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
        self.print_ss_stuff = print_ss_stuff

        # keep track of SmartStartPathing vs. NormalAgentPathing
        self.smart_start_pathing = False
        self.smart_start_path = None  # placeholder for smart_start_state to navigate to

        # the agent for navigating to the smartstart
        #TODO optimally you have preset parameters for each environment
        self.nnd_mb_agent = NND_MB_agent(env, sess,
                                         replay_buffer=self.replay_buffer,

                                         final_steps=nnd_mb_final_steps,
                                         steps_per_waypoint=nnd_mb_steps_per_waypoint,
                                         mean_per_stepsize=nnd_mb_mean_per_stepsize,
                                         std_per_stepsize=nnd_mb_std_per_stepsize,
                                         stepsizes_in_waypoint_radii=nnd_mb_stepsizes_in_waypoint_radii,

                                         gamma=nnd_mb_gamma,
                                         horizontal_penalty_factor=nnd_mb_horizontal_penalty_factor,
                                         horizon=nnd_mb_horizon,
                                         num_control_samples=nnd_mb_num_control_samples,
                                         path_shortcutting=nnd_mb_path_shortcutting,
                                         steps_before_giving_up_on_waypoint=nnd_mb_steps_before_giving_up_on_waypoint,

                                         save_dir_name=nnd_mb_save_dir_name,
                                         load_dir_name=nnd_mb_load_dir_name,
                                         save_training_data=nnd_mb_save_training_data,
                                         save_resulting_dynamics_model=nnd_mb_save_resulting_dynamics_model,
                                         load_existing_training_data=nnd_mb_load_existing_training_data,
                                         load_existing_dynamics_model=nnd_mb_load_existing_dynamics_model,

                                         num_fc_layers=nnd_mb_num_fc_layers,
                                         depth_fc_layers=nnd_mb_depth_fc_layers,
                                         batchsize=nnd_mb_batchsize,
                                         lr=nnd_mb_lr,
                                         nEpoch=nnd_mb_nEpoch,
                                         fraction_use_new=nnd_mb_fraction_use_new,
                                         num_episodes_for_aggregation=nnd_mb_num_episodes_for_aggregation,
                                         make_aggregated_dataset_noisy=nnd_mb_make_aggregated_dataset_noisy,
                                         make_training_dataset_noisy=nnd_mb_make_training_dataset_noisy,
                                         noise_actions_during_MPC_rollouts=nnd_mb_noise_actions_during_MPC_rollouts,

                                         verbose=nnd_mb_verbose,

                                         use_threading=nnd_mb_use_threading,
                                         num_rollouts_train=nnd_mb_num_rollouts_train,
                                         num_rollouts_val=nnd_mb_num_rollouts_val,
                                         dt_steps=nnd_mb_dt_steps,
                                         steps_per_rollout_train=nnd_mb_steps_per_rollout_train,
                                         steps_per_rollout_val=nnd_mb_steps_per_rollout_val)

        self.times_for_smart_start = []

    def get_param_dict(self):
        return self.param_dict

    def get_summary_name(self):
        """
        This is just for smartstart.reinforcementLearningCore.rlTrain.rlTrain
        :return: SmartStart string including the name of the base agent
        """
        if hasattr(self.agent, 'get_summary_name'):
            return "SmartStartC_" + self.agent.get_summary_name()
        else:
            return "SmartStartC_" + self.agent.__class__.__name__

    @property
    def normal_agent_pathing(self):
        return not self.smart_start_pathing

    def reduce_eta(self):
        """
        Reduces the chance of smart start happening (eta) (should happen after each episode)
        """
        self.eta = self.eta * self.eta_decay_factor

    def get_smart_start_path(self):
        """Determines the smart start state

        The smart start is determined using the UCB1 algorithm. The UCB1
        algorithm is a well known exploration strategy for multi-arm
        bandit problems. The smart start is chosen according to

        smart_start = \arg\max\limits_s\left(alpha * \max\limits_a Q(s,
        a) + \sqrt{\frac{beta * \log |D| }{C(s}} \right)

        Where
            * \alpha = exploitation_param
            * \beta  = exploration_param

        \max\limits_a Q(s,a) corresponds to V(s) which is determined by the agent

        |D| is the size of the replay buffer

        C(s) is the "visitation count" of the state. This is estimated through KDE #TODO scipy link
        KDE uses gaussian functions (could be other functions) around each given point, to create an estimated
        probability density function. This probability density estimates how likeley (if you were to stop the robot randomly)
        the robot would be at that state. To create a "count", we find the probability of a given state, by
        "integrating" over a single stepsize volume (volume of a hyperellipsoid where each radii is the average stepsize in that direction)
        (intrgration is approximate, we just take PDF at the state point, and multiply by volume)
        Then with the probability for that state, we multiply by the total number of states (|D|) to get a count

        Returns: Smart Start Path (list of states with the final state being the smart start state)
        """
        if len(self.replay_buffer) == 0: # if buffer is emtpy, nothing to evaluate
            return None
        possible_start_indices = self.replay_buffer.get_possible_smart_start_indices(self.n_ss)
        if possible_start_indices is None: # no valid states then return None
            return None
        #find the smart_start state
        all_states = self.replay_buffer.get_all_states()  # n x d matrix where n is the number of states and d is dim

        ##################### KERNEL CALCULATIONS AND ELLIPSOID VOLUME ############################
        kernel = scipy.stats.gaussian_kde(all_states.T, bw_method='scott') #TODO options for what type of bandwith calc
        # plot_2d_density(all_states.T[0], all_states.T[1], kernel) #TODO remove plot
        if self.nnd_mb_agent.radii is not None:
            one_radii_volume = volume_of_n_dimensional_hyperellipsoid(self.nnd_mb_agent.radii)
            # for a n-dimensional state vector, self.nn_mb_agent.radii is a n- long vector which each
            # value represents some combination of the mean and std of that index per step
            # the volume is the volume of a hyperellipsoid with those radii
        else:
            one_radii_volume = 1  # 100% arbitrary TODO: get std of last path maybe for both of these

        ################### PARALLEL UCB CALC #########################################################
        # t1 = time.time()
        possible_ss_steps = np.array(self.replay_buffer.buffer)[possible_start_indices]
        possible_ss_states = np.asarray(self.replay_buffer.steps_to_s2(possible_ss_steps).tolist()) #use tolist, because it renders as a numpy array of objects (not of floats)
        ss_state_values = self.agent.get_state_value(possible_ss_states).T # 1 x n matrix (equiv n long list)
        probability_densities = (kernel(possible_ss_states.T) * one_radii_volume) # 1 x n matrix (equiv n long list)
        C_hats = len(self.replay_buffer) * probability_densities
        ucb_list = self.exploitation_param * ss_state_values + \
              np.sqrt((self.exploration_param *
                       np.log(len(self.replay_buffer))) / C_hats)
        smart_start_parallel_index = possible_start_indices[np.argmax(ucb_list)]

        # ######### For loop setup #####################################################################
        # t2 = time.time()
        # smart_start_index = None
        # max_ucb = -float('inf')
        #
        # for main_step_index in possible_start_indices:
        #     # state value
        #     main_step = self.replay_buffer.buffer[main_step_index]
        #     state = self.replay_buffer.step_to_s2(main_step)
        #     state_value = self.agent.get_state_value(state)[0][0]
        #
        #     #SCIPY Kernel Density Estimation TODO document what math was used and what resources
        #     probability_density = (kernel(state.T) * one_radii_volume)[0]
        #     C_hat = len(all_states) * probability_density
        #
        #     #ucb calculation
        #     ucb = self.exploitation_param * state_value + \
        #           np.sqrt((self.exploration_param *
        #                    np.log(len(self.replay_buffer))) / C_hat)
        #     if ucb > max_ucb:
        #         smart_start_index = main_step_index
        #         max_ucb = ucb
        # print("Parallel took: " + str(t2 - t1) + "      |Iterative took: " + str(time.time() - t2) + " | " + str(smart_start_index == smart_start_parallel_index))
        return self.replay_buffer.get_episodic_path_to_buffer_index(smart_start_parallel_index)

    def get_action(self, state):
        """
        Either asks the Neural Network Dynamics model to choose action (Smart STart)
        or just lets the base agent go
        :param state: State currently at
        :return: action
        """
        if self.smart_start_pathing:
            return self.nnd_mb_agent.get_action(state)
        else:
            return self.agent.get_action(state)

    def observe(self, state, action, reward, new_state, done):
        """
        Record observation to replay buffer,
        Allow the base agent to observe the transition
        If smart start pathing is on, check if the goal is reached
        :return: Nothing
        """
        # new_state added (maybe) to buffer
        self.replay_buffer.add(self, state, action, reward, done, new_state)

        # agent observation
        self.agent.observe(state, action, reward, new_state, done)

        # check if smart_start_state has been reached
        if self.smart_start_pathing:
            self.nnd_mb_agent.observe(state, action, reward, new_state, done)
            if self.nnd_mb_agent.close_enough_to_goal(new_state):
                self.smart_start_pathing = False
                if self.print_ss_stuff:
                    print("distance to goal: " + str(self.nnd_mb_agent.distance_function(new_state,self.smart_start_path[-1])))
                    print("END OF SMART START STUFFS")

    def start_new_episode(self, state):
        """
        FIRST reset smart start pathing stuffs
        Then check if smart start will randomly happen, if so set it up (don't return)
        Finally tell the base_agent that a new episode is happening
        IMPORTANT: tell the replay buffer that a new episode is starting
            (this allows the replay buffer to keep track of where episodes begin/end which is necessary in retrieving smart start PATHS)
        :return: NOTHING
        """
        self.smart_start_pathing = False
        self.smart_start_path = None

        if np.random.rand() <= self.eta: #eta is probability of using smartStart
            start_time = time.time()
            self.smart_start_path = self.get_smart_start_path() # new state to navigate to
            end_time = time.time()
            if self.smart_start_path: #ensure path exists
                if self.print_ss_stuff:
                    elapsed_time = end_time - start_time
                    print("Calculate Smart Start Path Time: " + str(elapsed_time), end='')
                    print("\npath exists")
                # let neural network dynamics model based controller load the path
                self.nnd_mb_agent.start_new_episode_plan(state, self.smart_start_path)
                if not self.nnd_mb_agent.close_enough_to_goal(state): #ensure goal hasn't already been reached
                    self.smart_start_pathing = True #this start smart start navigation
                    if self.print_ss_stuff:
                        print("SMART_START START!!!")

        self.agent.start_new_episode(state)
        self.replay_buffer.start_new_episode(self)

    def end_episode(self):
        self.reduce_eta() #reduces the change of smart start
        self.agent.end_episode()
        self.smart_start_pathing = False
        self.smart_start_path = None

    def render(self, env, **kwargs):
        return env.render()

import time

parser = argparse.ArgumentParser(description='Set some flages')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--noGpu', action='store_true', help='If included, stops the gpu from being used')
parser.add_argument('-s', '--seed', dest="seed", default=int(time.time() * 10) % (2 ** 32 - 1), type=int)
args = parser.parse_args()
noGpu = args.noGpu
RANDOM_SEED = args.seed

if __name__ == "__main__":
    import random
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain, rlTrainGraphSS

    episodes = 1000
    lastLayerTanh = True

    # configuring environment
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    if noGpu:
        tfConfig = tf.ConfigProto(device_count={'GPU': 0})
    else:
        tfConfig = None

    with tf.Graph().as_default() as graph:
        with tf.Session(config=tfConfig, graph=graph) as sess:
            # Reset the seed for random number generation
            set_global_seeds(RANDOM_SEED)
            env.seed(RANDOM_SEED)

            # Initialize agent, see class for available parameters
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

            smart_start_agent = SmartStartContinuous(base_agent, env, sess,

                                                     buffer_size=100000,
                                                     exploitation_param=1.,
                                                     exploration_param=1.,
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
                                     plot_ss_stuff=False,
                                     print_time=True)

            noGpu_str = "-NoGPU" if noGpu else ""
            llTanh_str = "-LLTanh" if lastLayerTanh else ""
            summary.add_params_to_param_dict(zz_RANDOM_SEED=RANDOM_SEED, zz_episodes=episodes, noGpu=noGpu)
            fp = summary.save(get_default_data_directory("smart_start_continuous_summaries/0/"),
                              extra_name_append="-" + str(
                                  episodes) + "ep" + noGpu_str + "-noNorm" + llTanh_str + "-decayingNoise" + "-2000n_ss")

            train_writer = tf.summary.FileWriter(fp[:-5])
            train_writer.add_graph(sess.graph)
