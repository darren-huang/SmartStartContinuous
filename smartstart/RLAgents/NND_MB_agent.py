# imports
import os
import time

import numpy as np
import numpy.random as npr
import tensorflow as tf

# noinspection PyPackageRequirements,PyPackageRequirements
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.data_manipulation import \
    from_observation_to_usablestate, generate_training_data_inputs, generate_training_data_outputs
from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.dynamics_model import Dyn_Model
from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.helper_funcs import add_noise, perform_rollouts
from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.policy_random import Policy_Random
from smartstart.reinforcementLearningCore.agents_abstract_classes import NavigationRLAgent
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc, dist_line_seg_to_point, \
    elliptical_euclidean_distance_scaled_function_generator, path_shortcutter
from smartstart.utilities.utilities import get_start_waypoints_final_states_steps, get_default_model_directory
from smartstart.utilities.numerical import elliptical_euclidean_distance_function_generator


# def make_save_directories(run_num):
#     save_dir = 'run_' + str(run_num)
#     save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#         os.makedirs(save_dir + '/losses')
#         os.makedirs(save_dir + '/models')
#         # os.makedirs(save_dir + '/saved_forwardsim')
#         os.makedirs(save_dir + '/saved_trajfollow')
#         os.makedirs(save_dir + '/training_data')
#     return save_dir

def make_directories(dir_name):
    save_dir = get_default_model_directory(dir_name, create=False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        # os.makedirs(save_dir + '/saved_forwardsim')
        os.makedirs(save_dir + '/saved_trajfollow')
        os.makedirs(save_dir + '/training_data')
    return save_dir


class NND_MB_agent(NavigationRLAgent):  # Neural Network Dynamics Model Based Agent (NND_MB_agent)
    model_subdirectory_name = "NND_MB_agent"
    tf_datatype = tf.float64
    noiseToSignal = 0.01

    # n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
    actions_ag = 'nc'

    def __init__(self, env, sess,

                 replay_buffer=None, BUFFER_SIZE=10000,

                 final_steps=10, steps_per_waypoint=1, mean_per_stepsize=1, std_per_stepsize=1,
                 stepsizes_in_waypoint_radii=1,

                 gamma=.75, horizontal_penalty_factor=.5, horizon=20, num_control_samples=5000, path_shortcutting=True,
                 steps_before_giving_up_on_waypoint = 5,

                 save_dir_name="save_untitled", load_dir_name="untitled_load",
                 save_training_data=False, save_resulting_dynamics_model=False,
                 load_existing_training_data=False, load_existing_dynamics_model=False,

                 num_fc_layers=1, depth_fc_layers=500, batchsize=512, lr=0.001, nEpoch=30, fraction_use_new=0.9,
                 num_episodes_for_aggregation=3, make_aggregated_dataset_noisy=True,
                 make_training_dataset_noisy=True, noise_actions_during_MPC_rollouts=True,

                 verbose=True,

                 use_threading=True, num_rollouts_train=25, num_rollouts_val=20, dt_steps=3,
                 steps_per_rollout_train=333, steps_per_rollout_val=333):
        """
        ##################################    REQUIRED AGENT SETUP PARAMETERS       ##########################################
        :param env: the environment the agent is going to navigate
        :param sess: the tensorflow session

        ##################################    REPLAY BUFFER         ###########################################################
        :param replay_buffer: the buffer that stores previous experiences (state, action, reward, terminal, next_state) tuples
        :param BUFFER_SIZE: max size of the memory buffer

        ##################################    WAYPOINT STUFFS         ##########################################################
        :param final_steps: Once the Final waypoint is the 'next waypoint', the agent is given 'final_steps' number of
                            steps, after which if the waypoint isn't reached, the agent's self.close_enough_to_goal will return True
        :param steps_per_waypoint: the number of steps in between all waypoints
        :param mean_per_stepsize: given the mean length of a step, the number of means we want to include in our definition of 1 "step"
        :param std_per_stepsize: same as mean per stepsize, but with standard deviation of the step
        :param stepsizes_in_waypoint_radii: the number of "stepsizes" we want the size of our waypoint to be

        ##################################    MODEL PREDICTIVE CONTROLLER SETTINGS          #######################################
        :param gamma: when calculating the reward for the action sequences, gamma devalues the reward later actions
            ie. for 10 actions, the reward for the 2nd action is reward*gamma, 3rd action is reward*(gamma ** 2)
        :param horizontal_penalty_factor: when calculating the reward for an action sequence, there is a penalty for how
            far horizontally it is from the current line segement. This factor can reduce/increase that penalty
        :param horizon: Path following generates 'num_control_samples' amount of random action sequences that all 'horizon' long,
                        the best sequence will be selected (only first action is taken)
        :param num_control_samples: Path following generates this many random sequences of actions (each sequence is 'horizon' long)
                                    Controller will choose the action sequence with the best reward
        :param path_shortcutting: given a path to follow, whether or not to try to take shortcuts (based on numerical.py's 'path_shortcutter')
        :param steps_before_giving_up_on_waypoint: when tracking a path, after doing this many steps and not advancing to the next waypoint,
                                                    the agent will give up on current waypoint and move onto the next one (if it exists)

        ##################################    SAVING/LOADING STUFFS       #########################################################
        :param save_dir_name: the name of the directory the model will be saved
        :param load_dir_name: the name of the directory the model will be loaded
        :param save_training_data: whehter or not to save the training data
        :param save_resulting_dynamics_model: whether or not to save the dynamics model
        :param load_existing_training_data: whether or not to load the training data
        :param load_existing_dynamics_model: whether or not to load the dynamics model

        ##################################    NEURAL NETWORK DYNAMICS MODEL STUFFS    ############################################
        :param num_fc_layers: Number of fully connected layers in Neural Network Dynamics Model
        :param depth_fc_layers: Depth of fully connected layers in Neural Network Dynamics Model
        :param batchsize: The training batchsize of the Neural Network Dynamics Model
        :param lr: Learning rate of the Neural Network Dynamics model
        :param nEpoch: Number of times the Neural Network Dynamics model will train itself during the training sessions
        :param fraction_use_new: Model Training uses this percentage of replay_buffer data (rest is just training data)
        :param num_episodes_for_aggregation: After this many episodes, the agent will Aggregate the data, and train the model
        :param make_aggregated_dataset_noisy: Training uses old data (training data) and data inside the replay_buffer,
                                            the replay buffer data, is the aggregated dataset, and we can choose to make it noisy
        :param make_training_dataset_noisy: Whether or not to make training set noisy
        :param noise_actions_during_MPC_rollouts: Whether or not to put noise into the chosen actions

        ##################################    VERBOSITY     ########################################################################
        :param verbose: Print out statements

        ##################################    TRAINING DATA GATHERING      ###########################################################
        :param use_threading: whether or not to use threading for the data gathering
        :param num_rollouts_train: number of rollouts for the training data set
        :param num_rollouts_val: number of rollouts for the validation data set
        :param dt_steps: (UNSURE has to do with how data is collected)
        :param steps_per_rollout_train: When collecting training data, determines how much training data is collected (number of steps per simulation rollout)
        :param steps_per_rollout_val: When collecting data, determines how much validation data is collected (number of steps per simulation rollout)



        """

        theta = 1 # the minimum distance for the agent to consider a waypoint "reached" - distance defined by distance function
                  #BUT, now instead of changing the distance, we have changed the distance function
                  # therefore theta is always 1, but the distance function changes

        ### Initial Variables ###########################################################################
        self.final_steps = final_steps
        self.gamma = gamma
        self.horizontal_penalty_factor = horizontal_penalty_factor
        self.env = env
        self.N = num_control_samples
        self.horizon = horizon
        self.steps_per_waypoint = steps_per_waypoint
        self.mean_per_stepsize = mean_per_stepsize
        self.std_per_stepsize = std_per_stepsize
        self.stepsizes_in_waypoint_radii = stepsizes_in_waypoint_radii
        self.theta = theta  # minimum distance to get to desired state(s)
        self.use_existing_dynamics_model = load_existing_dynamics_model
        self.make_aggregated_dataset_noisy = make_aggregated_dataset_noisy
        self.nEpochs = nEpoch
        self.fraction_use_new = fraction_use_new
        self.num_episodes_for_aggregation = num_episodes_for_aggregation
        self.path_shortcutting = path_shortcutting
        self.steps_before_giving_up_on_waypoint = steps_before_giving_up_on_waypoint
        self.num_episodes_finished = 0
        self.actions_done_for_current_waypoint = None
        self.radii = None
        self.distance_function = None
        self.stds = None
        self.path_to_follow = None
        self.desired_states = None
        self.current_desired_state_index = None  # will start at 0 when a new episode begins otherwise error
        self.distances_left = None  # self.distances_left[x] will tell the distance from
        #                             desired_states[x] to desired_states[x+1] + [x+1] to [x+2].... until the end
        self.save_resulting_dynamics_model = save_resulting_dynamics_model

        self.dt_from_xml = 0  # env.env.model.opt.timestep  # TODO: dt_from_xml seems to be only for rendering

        # BY DEFAULT, directories are under .../models/NND_MB_agent
        self.load_dir = make_directories(os.path.join(self.model_subdirectory_name, load_dir_name))  ### make directories for loading data ###

        if save_resulting_dynamics_model or save_training_data:
            self.save_dir = make_directories(os.path.join(self.model_subdirectory_name, save_dir_name))  ### make directories for saving data ###
        else:
            self.save_dir = None

        if (noise_actions_during_MPC_rollouts):
            self.noise_amount = 0.005
        else:
            self.noise_amount = 0

        self.sess = sess

        # Initialize replay memory
        if replay_buffer == None:
            self.replay_buffer = ReplayBuffer(self, BUFFER_SIZE)  # type: ReplayBuffer
        else:
            self.replay_buffer = replay_buffer  # type: ReplayBuffer

        ### Get Training Data  ##########################################################################
        if (load_existing_training_data):
            if (verbose):
                print("\n#####################################")
                print("Retrieving training data & policy from saved files")
                print("#####################################\n")

            self.dataX = np.load(self.load_dir + '/training_data/dataX.npy')  # input1: state
            self.dataY = np.load(self.load_dir + '/training_data/dataY.npy')  # input2: control
            self.dataZ = np.load(self.load_dir + '/training_data/dataZ.npy')  # output: nextstate-state
            self.states_val = np.load(self.load_dir + '/training_data/states_val.npy')
            self.controls_val = np.load(self.load_dir + '/training_data/controls_val.npy')
            # self.forwardsim_x_true = np.load(self.load_dir + '/training_data/forwardsim_x_true.npy')
            # self.forwardsim_y = np.load(self.load_dir + '/training_data/forwardsim_y.npy')
        else:
            random_policy = Policy_Random(env)  # create random policy for data collection

            # data collection, either with or without multi-threading
            if (use_threading):
                from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.collect_samples_threaded import CollectSamples
            else:
                from smartstart.RLContinuousAlgorithms.NN_Dynamics_Model.collect_samples import CollectSamples

            if (verbose):
                print("\n#####################################")
                print("Performing rollouts to collect training data")
                print("#####################################\n")

            # perform rollouts`
            states, controls, _, _ = perform_rollouts(random_policy, num_rollouts_train, steps_per_rollout_train,
                                                      False, CollectSamples, env, dt_steps, self.dt_from_xml)
            states = np.array(states)

            if (verbose):
                print("\n#####################################")
                print("Performing rollouts to collect validation data")
                print("#####################################\n")

            start_validation_rollouts = time.time()
            self.states_val, self.controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val,
                                                                        steps_per_rollout_val, False,
                                                                        CollectSamples, env, dt_steps, self.dt_from_xml)
            self.states_val = np.array(self.states_val)

            # if (verbose):
            #     print("\n#####################################")
            #     print("Convert from env observations to NN 'states' ")
            #     print("#####################################\n")

            # DH - Don't think we need to change the observed state, maybe later
            # training
            # states = from_observation_to_usablestate(states, which_agent, False)
            # validation
            # self.states_val = from_observation_to_usablestate(self.states_val, which_agent, False)

            if (verbose):
                print("\n#####################################")
                print("Data formatting: create inputs and labels for NN ")
                print("#####################################\n")

            self.dataX, self.dataY = generate_training_data_inputs(states, controls)
            self.dataZ = generate_training_data_outputs(states)

            if (verbose):
                print("\n#####################################")
                print("Add noise")
                print("#####################################\n")

            # add a little dynamics noise (next state is not perfectly accurate, given correct state and action)
            if (make_training_dataset_noisy):
                self.dataX = add_noise(self.dataX, self.noiseToSignal)
                self.dataZ = add_noise(self.dataZ, self.noiseToSignal)

            if (verbose):
                print("\n#####################################")
                print("Perform rollout & save for forward sim")
                print("#####################################\n")

            # states_forwardsim_orig, controls_forwardsim, _, _ = perform_rollouts(random_policy, 1, 100, False,
            #                                                                      CollectSamples, env, dt_steps,
            #                                                                      self.dt_from_xml)
            # states_forwardsim = np.copy(from_observation_to_usablestate(states_forwardsim_orig, which_agent, False))
            # self.forwardsim_x_true, self.forwardsim_y = generate_training_data_inputs(states_forwardsim,
            #                                                                           controls_forwardsim)

            if (verbose):
                print("\n#####################################")
                print("Saving data")
                print("#####################################\n")

            if save_training_data:
                np.save(self.save_dir + '/training_data/dataX.npy', self.dataX)
                np.save(self.save_dir + '/training_data/dataY.npy', self.dataY)
                np.save(self.save_dir + '/training_data/dataZ.npy', self.dataZ)
                np.save(self.save_dir + '/training_data/states_val.npy', self.states_val)
                np.save(self.save_dir + '/training_data/controls_val.npy', self.controls_val)
                # np.save(self.save_dir + '/training_data/forwardsim_x_true.npy', self.forwardsim_x_true)
                # np.save(self.save_dir + '/training_data/forwardsim_y.npy', self.forwardsim_y)

        # XYZ variable data #############################################################################
        # every component (i.e. x position) should become mean 0, std 1
        self.mean_x = np.mean(self.dataX, axis=0)
        self.dataX = self.dataX - self.mean_x
        self.std_x = np.std(self.dataX, axis=0)
        self.dataX = np.nan_to_num(self.dataX / self.std_x)

        self.mean_y = np.mean(self.dataY, axis=0)
        self.dataY = self.dataY - self.mean_y
        self.std_y = np.std(self.dataY, axis=0)
        self.dataY = np.nan_to_num(self.dataY / self.std_y)

        self.mean_z = np.mean(self.dataZ, axis=0)
        self.dataZ = self.dataZ - self.mean_z
        self.std_z = np.std(self.dataZ, axis=0)
        self.dataZ = np.nan_to_num(self.dataZ / self.std_z)

        ## concatenate state and action, to be used for training dynamics
        self.inputs = np.concatenate((self.dataX, self.dataY), axis=1)
        self.outputs = np.copy(self.dataZ)

        # Create Dynamics Model #########################################################################
        # dimensions
        assert self.inputs.shape[0] == self.outputs.shape[0]
        inputSize = self.inputs.shape[1]
        outputSize = self.outputs.shape[1]

        # initialize dynamics model
        self.dyn_model = Dyn_Model(inputSize, outputSize, self.sess, lr, batchsize, num_fc_layers, depth_fc_layers,
                                   self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z,
                                   self.tf_datatype, verbose)

        # randomly initialize all vars
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)

    def get_action(self, state):
        return self.get_action_with_predicted_states(state)[0]

    def get_action_with_predicted_states(self, state):
        self.actions_done_for_current_waypoint += 1
        best_action, best_sim_number, best_sequence, best_path = self.get_best_sim_actions(state)

        # whether to execute noisy or clean actions
        noise_actions = True
        if (self.actions_ag == 'nn'):
            noise_actions = True
        if (self.actions_ag == 'nc'):
            noise_actions = True
        if (self.actions_ag == 'cc'):
            noise_actions = False

        # noise the action
        action_to_take = np.copy(best_action)
        if (noise_actions):
            noise = self.noise_amount * npr.normal(size=action_to_take.shape)  #
            action_to_take = action_to_take + noise  # no clip maybe go over bounds for action space

        return action_to_take, best_path

    def observe(self, state, action, reward, new_state, done):
        # get distances to current and next waypoints
        self.replay_buffer.add(self, state, action, reward, done, new_state)

        distance_to_current = self.distance_function(new_state, self.current_desired_state)
        distance_to_next = self.distance_function(new_state, self.next_desired_state)

        # TODO decide which one to use or to use both (theta, or distances to next<=current)
        if self.move_to_next(new_state, self.current_desired_state_index, distance_to_current, distance_to_next) or \
                (self.actions_done_for_current_waypoint > self.steps_before_giving_up_on_waypoint and
                 self.current_desired_state_index != len(self.desired_states) - 1): #if spending too long just give up
            # close enough to curr_waypoint, move on to next waypoint
            self.current_desired_state_index += 1
            self.actions_done_for_current_waypoint = 0

    def start_new_episode_plan(self, starting_state, path_to_follow):
        """
        Called at the beginning of an episode.
        :param starting_state: the starting state of the episode
        :param path_to_follow: the states it wants to navigate through, agent will try to follow
            route and end up within close proximity to the last state
        """
        # set reset variables for the new episode
        self.current_desired_state_index = 0
        self.actions_done_for_current_waypoint = 0


        # radii calc (based off of standard deviation and mean of step sizes)
        stds, means = path_deltas_stds_and_means_per_dim(path_to_follow)
        self.stds = stds
        self.radii = radii_calc(means, stds, self.mean_per_stepsize, self.std_per_stepsize,
                                self.stepsizes_in_waypoint_radii)

        # set the distance function to have everything on the ellipse defined by 'radii' to be exactly a distance of 1
        self.distance_function = \
            elliptical_euclidean_distance_function_generator(self.radii)

        #shorten path if necessary (note: radii calculations will be based off of given path, not shortcutted path)
        if self.path_shortcutting:
            self.path_to_follow = path_shortcutter(path_to_follow, self.distance_function, self.theta)
        else:
            self.path_to_follow = path_to_follow
        # ADDING EXTRA WAYPOINTS, doesn't really help
        # dist_to_start = self.distance_function(starting_state, self.path_to_follow[0])
        # extra_waypoints = int(np.ceil(dist_to_start))
        # delta = (self.path_to_follow[0] - np.array(starting_state))/extra_waypoints
        # self.path_to_follow = np.vstack(([(starting_state + (delta * i)).tolist() for i in range(extra_waypoints)],self.path_to_follow))
        desired_states = np.asarray(get_start_waypoints_final_states_steps(self.path_to_follow, self.steps_per_waypoint))
        self.desired_states = desired_states

        # calculate distances for the MPC reward function
        if len(desired_states) >= 2:
            temporal_differences = \
                [self.distance_function(desired_states[x - 1], desired_states[x]) for x in
                 range(1, len(desired_states))] + [0]
            self.distances_left = [sum(temporal_differences[i:]) for i in range(len(desired_states))]
        else:
            self.distances_left = [0]
        self.distances_left = np.asarray(self.distances_left)

        # train
        if self.num_episodes_finished % self.num_episodes_for_aggregation == 0:
            self.train_dynamics_model()
        self.num_episodes_finished += 1

    def close_enough_to_goal(self, current_state):
        if self.distance_function(current_state, self.desired_states[-1]) <= self.theta:
            return True
        # TODO make this bettter
        if self.current_desired_state_index == len(self.desired_states) - 1 and \
                self.final_steps <= self.actions_done_for_current_waypoint:
            return True
        return False

    def render(self, env, **kwargs):
        env.render()

    def train_dynamics_model(self):
        """
        Feeds the aggregation data along with the initial training data to the neural net dynamics model
        """
        # s -> datax  || a -> datay || s2 -> dataz
        if len(self.replay_buffer) == 0:
            s_batch = np.zeros((0, self.dataX.shape[1]))
            a_batch = np.zeros((0, self.dataY.shape[1]))
            s2_batch = np.zeros((0, self.dataZ.shape[1]))
        else:
            s_batch, a_batch, _, _, s2_batch = self.replay_buffer.all_batch()
        new_dataX = s_batch
        new_dataY = a_batch
        new_dataZ = s2_batch - s_batch
        if self.make_aggregated_dataset_noisy:
            new_dataX = add_noise(new_dataX, self.noiseToSignal)
            new_dataZ = add_noise(new_dataZ, self.noiseToSignal)
        dataX_new_preprocessed = np.nan_to_num((new_dataX - self.mean_x) / self.std_x)
        dataY_new_preprocessed = np.nan_to_num((new_dataY - self.mean_y) / self.std_y)
        dataZ_new_preprocessed = np.nan_to_num((new_dataZ - self.mean_z) / self.std_z)

        ## concatenate state and action, to be used for training dynamics
        inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
        outputs_new = np.copy(dataZ_new_preprocessed)

        # train model or restore model
        if (self.use_existing_dynamics_model):
            restore_path = self.load_dir + '/models/finalModel.ckpt'
            self.saver.restore(self.sess, restore_path)
            print("Model restored from ", restore_path)
            training_loss = 0
            old_loss = 0
            new_loss = 0
        else:
            training_loss, old_loss, new_loss = self.dyn_model.train(self.inputs, self.outputs,
                                                                     inputs_new, outputs_new,
                                                                     self.nEpochs, self.save_dir,
                                                                     self.fraction_use_new,
                                                                     save_results=self.save_resulting_dynamics_model)
        if self.save_resulting_dynamics_model:
            save_path = self.saver.save(self.sess, self.save_dir + '/models/model_numTrain' +
                                        str(1 + (
                                                self.num_episodes_finished // self.num_episodes_for_aggregation)) + '.ckpt')
            save_path = self.saver.save(self.sess, self.save_dir + '/models/finalModel.ckpt')

    @property
    def current_desired_state(self):
        return self.desired_states[self.current_desired_state_index]

    @property
    def next_desired_state(self):
        return self.desired_states[min(self.current_desired_state_index + 1, len(self.desired_states) - 1)]

    # private
    def move_to_next(self, pt, desired_state_index, distance_to_curr, distance_to_next):
        move_to_next = np.logical_and(
            # distance_to_curr <= self.theta,
            np.logical_or(distance_to_curr <= self.theta, distance_to_next <= distance_to_curr),
            desired_state_index != len(self.desired_states) - 1)
        return move_to_next

    def get_best_sim_actions(self, curr_nn_state):
        # randomly sample N candidate action sequences
        all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high,
                                  (self.N, self.horizon, self.env.action_space.shape[0]))

        # forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        resulting_states = self.dyn_model.do_forward_sim([curr_nn_state, 0], np.copy(all_samples), many_in_parallel)
        resulting_states = np.array(resulting_states)  # this is [horizon+1, N, statesize]

        # moved_to_next = np.zeros((self.N,))
        # done_forever = np.zeros((self.N,))
        # prev_forward = np.zeros((self.N,))
        # prev_pt = resulting_states[0]

        # scores, best_score, best_sim_number = self.generate_scores_add_total_dist(resulting_states)
        scores, best_score, best_sim_number = self.generate_scores_add_delta(resulting_states)

        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])
        best_path = resulting_states[:, best_sim_number]

        return best_action, best_sim_number, best_sequence, best_path

    def generate_scores_add_total_dist(self, resulting_states):
        # init vars to evaluate the trajectories
        scores = np.zeros((self.N,))

        samples_desired_state_indices = np.tile(self.current_desired_state_index, (self.N,))
        samples_desired_state_indices = samples_desired_state_indices.astype(int)

        distances_to_end = np.zeros((self.N,))

        # accumulate reward over each timestep
        for pt_number in range(resulting_states.shape[0]):
            # array of "the point"... for each sim
            pt = resulting_states[pt_number]  # N x state

            # get distances to current and next waypoints
            current_desired_states = self.desired_states[samples_desired_state_indices]
            next_desired_states = self.desired_states[
                np.minimum(samples_desired_state_indices + 1, len(self.desired_states) - 1)]
            distances_to_current = self.distance_function(current_desired_states, pt)
            distances_to_next = self.distance_function(next_desired_states, pt)

            # boolean array, tells whether or not the sample moves to the next waypoint
            move_to_next = self.move_to_next(pt, samples_desired_state_indices, distances_to_current, distances_to_next)

            # update scores as the sum of distances left on path
            not_move_to_next = np.logical_not(move_to_next)
            distances_to_end[not_move_to_next] = \
                self.distances_left[samples_desired_state_indices[not_move_to_next]] + \
                distances_to_current[not_move_to_next]
            distances_to_end[move_to_next] = \
                self.distances_left[samples_desired_state_indices[move_to_next] + 1] + \
                distances_to_next[move_to_next]

            # update scores
            scores += distances_to_end

            # update next waypoint for each sample if necessary
            samples_desired_state_indices[move_to_next] += 1  # if within theta of waypoint, increment waypoint
        # pick best action sequence
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores)

        return scores, best_score, best_sim_number

    def generate_scores_add_delta(self, resulting_states):
        # init vars to evaluate the trajectories
        scores = np.zeros((self.N,))

        samples_desired_state_indices = np.tile(self.current_desired_state_index, (self.N,))
        samples_desired_state_indices = samples_desired_state_indices.astype(int)

        pts = resulting_states[0]  # starting state
        current_desired_states = self.desired_states[samples_desired_state_indices]
        prev_distances_to_end = self.distances_left[samples_desired_state_indices] + \
                                self.distance_function(pts, current_desired_states)

        distances_to_end = np.zeros((self.N,))

        # accumulate reward over each timestep
        for pt_number in range(resulting_states.shape[0]):
            # array of "the point"... for each sim
            pts = resulting_states[pt_number]  # N x state

            # get distances to current and next waypoints
            current_desired_states = self.desired_states[samples_desired_state_indices]
            next_desired_states = self.desired_states[
                np.minimum(samples_desired_state_indices + 1, len(self.desired_states) - 1)]
            distances_to_current = self.distance_function(current_desired_states, pts)
            distances_to_next = self.distance_function(next_desired_states, pts)

            # TO/DO remove:
            # if pt_number == 0:
            #     print("curr: " + str(distances_to_current[0]))
            #     print("next: " + str(distances_to_next[0]))

            # check if each sample should move onto the next waypoint
            move_to_next = self.move_to_next(pts, samples_desired_state_indices, distances_to_current,
                                             distances_to_next)
            samples_desired_state_indices[move_to_next] += 1  # if within theta of waypoint, increment waypoint

            # get distance to new/current waypoint (updated)
            distances_to_current[move_to_next] = distances_to_next[move_to_next]

            # calculated distance left on the path
            distances_to_end = self.distances_left[samples_desired_state_indices] + distances_to_current

            # update scores as the sum of distances left on path
            scores += (prev_distances_to_end - distances_to_end) * (self.gamma ** (pt_number))  # add the delta forward
            np.copyto(prev_distances_to_end, distances_to_end)

            # penalty for horizontal distance from the line segment
            # (line segment determined by the prev and current waypoint except for 1st waypoint)
            valid_penalties = samples_desired_state_indices >= 1  # needs a current and previous waypoint
            line_seg_begin_indices = np.maximum(samples_desired_state_indices - 1, 0)  # ensures no -1's
            dist = dist_line_seg_to_point(
                self.desired_states[line_seg_begin_indices],  # prev waypoint
                self.desired_states[line_seg_begin_indices + 1],  # current waypoint
                pts,
                self.distance_function,
                self.radii)
            scores -= dist * self.horizontal_penalty_factor * self.gamma

        # pick best action sequence
        best_score = np.max(scores)
        best_sim_number = np.argmax(scores)

        return scores, best_score, best_sim_number
