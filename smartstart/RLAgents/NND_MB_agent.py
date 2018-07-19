# imports
import os
import time
import re

import numpy as np
import numpy.random as npr
import tensorflow as tf

from data_manipulation import from_observation_to_usablestate
from data_manipulation import generate_training_data_inputs
from data_manipulation import generate_training_data_outputs
from dynamics_model import Dyn_Model
from helper_funcs import add_noise
from helper_funcs import perform_rollouts
# my imports
from policy_random import Policy_Random
# noinspection PyPackageRequirements,PyPackageRequirements
from smartstart.RLAgents.replay_buffer import ReplayBuffer
from smartstart.reinforcementLearningCore.agents import NavigationRLAgent
from smartstart.utilities.plot import plot_path, show_plot, ion_plot, pause_plot, update_path, ioff_plot
from smartstart.utilities.datacontainers import Summary, Episode
from smartstart.utilities.utilities import get_default_directory, get_start_waypoints_final_states_steps
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc, dist_line_seg_to_point

# from rllab.rllab.envs.normalized_env import normalize
from utilities.numerical import elliptical_euclidean_distance_function_generator


def make_save_directories(run_num):
    save_dir = 'run_' + str(run_num)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/saved_forwardsim')
        os.makedirs(save_dir + '/saved_trajfollow')
        os.makedirs(save_dir + '/training_data')
    return save_dir


class NND_MB_agent(NavigationRLAgent):  # Neural Network Dynamics Model Based Agent (NND_MB_agent)
    tf_datatype = tf.float64
    noiseToSignal = 0.01

    # n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
    actions_ag = 'nc'

    def __init__(self, env, sess, final_steps=10, replay_buffer=None, BUFFER_SIZE=10000, steps_per_waypoint=10,
                 mean_per_stepsize=1, std_per_stepsize=1, stepsizes_in_waypoint_radii=1., gamma=.75,
                 horizontal_penalty_factor=1, run_num=0, use_existing_training_data=False,
                 use_existing_dynamics_model=False, num_rollouts_save_for_mf=60, print_minimal=False, which_agent=2,
                 use_threading=True, num_rollouts_train=25, num_rollouts_val=20, num_fc_layers=1, depth_fc_layers=500,
                 batchsize=512, lr=0.001, nEpoch=30, fraction_use_new=0.9, horizon=20, num_control_samples=5000,
                 num_episodes_for_aggregation=10, rollouts_forTraining=9, make_aggregated_dataset_noisy=True,
                 make_training_dataset_noisy=True, noise_actions_during_MPC_rollouts=True, dt_steps=3,
                 steps_per_rollout_train=333, steps_per_rollout_val=333, visualize_False=False):
        """
        :param env: the environment the agent is going to navigate
        :param replay_buffer: the buffer that stores previous experiences (state, action, reward, terminal, next_state) tuples
        :param BUFFER_SIZE: max size of the memory buffer
        :param theta: the minimum distance for the agent to consider a waypoint "reached" - distance defined by distance function
        :param steps_per_waypoint: the number of steps in between all waypoints
        :param mean_per_stepsize: given the mean length of a step, the number of means we want to include in our definition of 1 "step"
        :param std_per_stepsize: same as mean per stepsize, but with standard deviation of the step
        :param stepsizes_in_waypoint_radii: the number of "stepsizes" we want the size of our waypoint to be

        :param run_num: the number that labels the run, determines the name of the folder to save/load to/from
        :param use_existing_training_data: whether or not to load the training data
        :param use_existing_dynamics_model: whether or not to load the dynamics model
        :param gamma: when calculating the reward for the action sequences, gamma devalues the reward later actions
            ie. for 10 actions, the reward for the 2nd action is reward*gamma, 3rd action is reward*(gamma ** 2)
        :param horizontal_penalty_factor: when calculating the reward for an action sequence, there is a penalty for how
            far horizontally it is from the current line segement. This factor can reduce/increase that penalty
        :param num_rollouts_save_for_mf:
        :param might_render:
        :param visualize_MPC_rollout:
        :param perform_forwardsim_for_vis:
        :param print_minimal:
        :param which_agent:
        :param use_threading:
        :param num_rollouts_train:
        :param num_rollouts_val:
        :param num_fc_layers:
        :param depth_fc_layers:
        :param batchsize:
        :param lr:
        :param nEpoch:
        :param fraction_use_new:
        :param horizon:
        :param num_control_samples:
        :param num_episodes_for_aggregation:
        :param rollouts_forTraining:
        :param make_aggregated_dataset_noisy:
        :param make_training_dataset_noisy:
        :param noise_actions_during_MPC_rollouts:
        :param dt_steps:
        :param steps_per_rollout_train:
        :param steps_per_rollout_val:
        :param visualize_False:
        """
        theta = 1

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
        self.use_existing_dynamics_model = use_existing_dynamics_model
        self.make_aggregated_dataset_noisy = make_aggregated_dataset_noisy
        self.nEpochs = nEpoch
        self.fraction_use_new = fraction_use_new
        self.num_episodes_for_aggregation = num_episodes_for_aggregation
        self.num_episodes_finished = 0
        self.actions_done_for_current_waypoint = None
        self.radii = None
        self.distance_function = None
        self.path_to_follow = None
        self.desired_states = None
        self.current_desired_state_index = None  # will start at 0 when a new episode begins otherwise error
        self.distances_left = None  # self.distances_left[x] will tell the distance from
        #                             desired_states[x] to desired_states[x+1] + [x+1] to [x+2].... until the end

        self.dt_from_xml = 0  # env.env.model.opt.timestep  # TODO: dt_from_xml seems to be only for rendering

        self.save_dir = make_save_directories(run_num)  ### make directories for saving data ###

        if (noise_actions_during_MPC_rollouts):
            self.noise_amount = 0.005
        else:
            self.noise_amount = 0

        self.sess = sess  # TODO: decide if i need GPU options

        # Initialize replay memory
        if replay_buffer == None:
            self.replay_buffer = ReplayBuffer(self, BUFFER_SIZE)  # type: ReplayBuffer
        else:
            self.replay_buffer = replay_buffer  # type: ReplayBuffer

        ### Get Training Data  ##########################################################################
        if (use_existing_training_data):
            if (not (print_minimal)):
                print("\n#####################################")
                print("Retrieving training data & policy from saved files")
                print("#####################################\n")

            self.dataX = np.load(self.save_dir + '/training_data/dataX.npy')  # input1: state
            self.dataY = np.load(self.save_dir + '/training_data/dataY.npy')  # input2: control
            self.dataZ = np.load(self.save_dir + '/training_data/dataZ.npy')  # output: nextstate-state
            self.states_val = np.load(self.save_dir + '/training_data/states_val.npy')
            self.controls_val = np.load(self.save_dir + '/training_data/controls_val.npy')
            self.forwardsim_x_true = np.load(self.save_dir + '/training_data/forwardsim_x_true.npy')
            self.forwardsim_y = np.load(self.save_dir + '/training_data/forwardsim_y.npy')
        else:
            random_policy = Policy_Random(env)  # create random policy for data collection

            # data collection, either with or without multi-threading
            if (use_threading):
                from collect_samples_threaded import CollectSamples
            else:
                from collect_samples import CollectSamples

            if (not (print_minimal)):
                print("\n#####################################")
                print("Performing rollouts to collect training data")
                print("#####################################\n")

            # perform rollouts`
            states, controls, _, _ = perform_rollouts(random_policy, num_rollouts_train, steps_per_rollout_train,
                                                      visualize_False, CollectSamples, env, dt_steps, self.dt_from_xml)
            states = np.array(states)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Performing rollouts to collect validation data")
                print("#####################################\n")

            start_validation_rollouts = time.time()
            self.states_val, self.controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val,
                                                                        steps_per_rollout_val, visualize_False,
                                                                        CollectSamples, env, dt_steps, self.dt_from_xml)
            self.states_val = np.array(self.states_val)

            # if (not (print_minimal)):
            #     print("\n#####################################")
            #     print("Convert from env observations to NN 'states' ")
            #     print("#####################################\n")

            # DH - Don't think we need to change the observed state, maybe later
            # training
            # states = from_observation_to_usablestate(states, which_agent, False)
            # validation
            # self.states_val = from_observation_to_usablestate(self.states_val, which_agent, False)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Data formatting: create inputs and labels for NN ")
                print("#####################################\n")

            self.dataX, self.dataY = generate_training_data_inputs(states, controls)
            self.dataZ = generate_training_data_outputs(states, which_agent)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Add noise")
                print("#####################################\n")

            # add a little dynamics noise (next state is not perfectly accurate, given correct state and action)
            if (make_training_dataset_noisy):
                self.dataX = add_noise(self.dataX, self.noiseToSignal)
                self.dataZ = add_noise(self.dataZ, self.noiseToSignal)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Perform rollout & save for forward sim")
                print("#####################################\n")

            states_forwardsim_orig, controls_forwardsim, _, _ = perform_rollouts(random_policy, 1, 100, visualize_False,
                                                                                 CollectSamples, env, dt_steps,
                                                                                 self.dt_from_xml)
            states_forwardsim = np.copy(from_observation_to_usablestate(states_forwardsim_orig, which_agent, False))
            self.forwardsim_x_true, self.forwardsim_y = generate_training_data_inputs(states_forwardsim,
                                                                                      controls_forwardsim)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Saving data")
                print("#####################################\n")

            np.save(self.save_dir + '/training_data/dataX.npy', self.dataX)
            np.save(self.save_dir + '/training_data/dataY.npy', self.dataY)
            np.save(self.save_dir + '/training_data/dataZ.npy', self.dataZ)
            np.save(self.save_dir + '/training_data/states_val.npy', self.states_val)
            np.save(self.save_dir + '/training_data/controls_val.npy', self.controls_val)
            np.save(self.save_dir + '/training_data/forwardsim_x_true.npy', self.forwardsim_x_true)
            np.save(self.save_dir + '/training_data/forwardsim_y.npy', self.forwardsim_y)

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
                                   self.tf_datatype, print_minimal)

        # randomly initialize all vars
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)

        # TODO: ROLLOUTS FOR TRAINING DATA!!!!!

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
        if self.move_to_next(new_state, self.current_desired_state_index, distance_to_current, distance_to_next):
            # close enough to curr_waypoint, move on to next waypoint
            self.current_desired_state_index += 1
            self.actions_done_for_current_waypoint = 0

    def start_new_episode_plan(self, starting_state, path_to_follow):
        """
        Called at the beginning of an episode.
        :param starting_state: the starting state of the episode
        :param desired_states: the states it wants to navigate through, agent will try to follow
            route and end up within close proximity to the last state
        """
        # set variables for the new episode
        self.current_desired_state_index = 0
        self.actions_done_for_current_waypoint = 0
        self.path_to_follow = path_to_follow
        desired_states = np.asarray(get_start_waypoints_final_states_steps(path_to_follow, self.steps_per_waypoint))
        self.desired_states = desired_states

        # radii calc (based off of standard deviation and mean of step sizes)
        stds, means = path_deltas_stds_and_means_per_dim(path_to_follow)
        self.radii = radii_calc(means, stds, self.mean_per_stepsize, self.std_per_stepsize, self.stepsizes_in_waypoint_radii)

        # set the distance function to have everything on the ellipse defined by 'radii' to be exactly a distance of 1
        self.distance_function = \
            elliptical_euclidean_distance_function_generator(self.radii)


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
        #TODO make this bettter
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
            restore_path = self.save_dir + '/models/finalModel.ckpt'
            self.saver.restore(self.sess, restore_path)
            print("Model restored from ", restore_path)
            training_loss = 0
            old_loss = 0
            new_loss = 0
        else:
            training_loss, old_loss, new_loss = self.dyn_model.train(self.inputs, self.outputs,
                                                                     inputs_new, outputs_new,
                                                                     self.nEpochs, self.save_dir,
                                                                     self.fraction_use_new)

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

    #private
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
        best_path = resulting_states[:,best_sim_number]

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

            #update scores
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

        pts = resulting_states[0] #starting state
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

            #TO/DO remove:
            # if pt_number == 0:
            #     print("curr: " + str(distances_to_current[0]))
            #     print("next: " + str(distances_to_next[0]))

            # check if each sample should move onto the next waypoint
            move_to_next = self.move_to_next(pts, samples_desired_state_indices, distances_to_current, distances_to_next)
            samples_desired_state_indices[move_to_next] += 1  # if within theta of waypoint, increment waypoint

            # get distance to new/current waypoint (updated)
            distances_to_current[move_to_next] = distances_to_next[move_to_next]

            # calculated distance left on the path
            distances_to_end = self.distances_left[samples_desired_state_indices] + distances_to_current

            # update scores as the sum of distances left on path
            scores += (prev_distances_to_end - distances_to_end) * (self.gamma ** (pt_number)) #add the delta forward
            np.copyto(prev_distances_to_end, distances_to_end)

            # penalty for horizontal distance from the line segment
            # (line segment determined by the prev and current waypoint except for 1st waypoint)
            valid_penalties = samples_desired_state_indices >= 1 # needs a current and previous waypoint
            line_seg_begin_indices = np.maximum(samples_desired_state_indices - 1, 0) #ensures no -1's
            dist = dist_line_seg_to_point(
                self.desired_states[line_seg_begin_indices], #prev waypoint
                self.desired_states[line_seg_begin_indices + 1], #current waypoint
                pts,
                self.radii)
            scores -= dist * self.horizontal_penalty_factor * self.gamma

        # pick best action sequence
        best_score = np.max(scores)
        best_sim_number = np.argmax(scores)

        #TODO remove following for loop:
        # best_path = resulting_states[:,best_sim_number]
        # sample_desired_state_index = int(self.current_desired_state_index)
        # pt = best_path[0]  # starting state
        # current_desired_state = self.desired_states[sample_desired_state_index]
        # prev_distance_to_end = self.distances_left[sample_desired_state_index] + \
        #                         self.distance_function(pt, current_desired_state)
        # distance_to_end = 0
        # score = 0
        # for pt_number in range(best_path.shape[0]):
        #     # array of "the point"... for each sim
        #     pt = best_path[pt_number]  # N x state
        #
        #     # get distances to current and next waypoints
        #     current_desired_state = self.desired_states[sample_desired_state_index]
        #     next_desired_state = self.desired_states[
        #         np.minimum(sample_desired_state_index + 1, len(self.desired_states) - 1)]
        #     distance_to_current = self.distance_function(current_desired_state, pt)
        #     distance_to_next = self.distance_function(next_desired_state, pt)
        #
        #     # check if each sample should move onto the next waypoint
        #     move_to_next = self.move_to_next(pt, sample_desired_state_index, distance_to_current,
        #                                      distance_to_next)
        #     if move_to_next:
        #         sample_desired_state_index += 1  # if within theta of waypoint, increment waypoint
        #         # get distance to new/current waypoint (updated)
        #         distance_to_current = distance_to_next
        #
        #     # calculated distance left on the path
        #     distance_to_end = self.distances_left[sample_desired_state_index] + distance_to_current
        #
        #     # update scores as the sum of distances left on path
        #     score += (prev_distance_to_end - distance_to_end) * (gamma ** (pt_number))  # add the delta forward
        #     prev_distance_to_end = distance_to_end
        #
        #     # penalty for horizontal distance from the line segment (line segment determined by the prev and current waypoint)
        #     if sample_desired_state_index >= 1:  # needs a current and previous waypoint
        #         dist = dist_line_seg_to_point(
        #             self.desired_states[sample_desired_state_index - 1],  # prev waypoint
        #             self.desired_states[sample_desired_state_index],  # current waypoint
        #             pt,
        #             self.radii) * horizontal_penalty_factor
        #         # print(dist)
        #         score -= dist
        # print("distance from point: " + str(self.distance_function(resulting_states[0][0],self.desired_states[0])))
        # print("'projection' from line segment: " +
        #       str(dist_line_seg_to_point(self.desired_states[0], self.desired_states[1],
        #                                  resulting_states[0][0], self.radii)))



        return scores, best_score, best_sim_number
