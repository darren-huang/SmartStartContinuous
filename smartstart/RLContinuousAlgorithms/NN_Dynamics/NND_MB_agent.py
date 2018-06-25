# imports
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import copy
import os
import sys
from six.moves import cPickle
# from rllab.rllab.envs.normalized_env import normalize
import yaml
import argparse
import json

# my imports
from policy_random import Policy_Random
from trajectories import make_trajectory
from trajectories import get_trajfollow_params
from data_manipulation import generate_training_data_inputs
from data_manipulation import generate_training_data_outputs
from data_manipulation import from_observation_to_usablestate
from data_manipulation import get_indices
from helper_funcs import perform_rollouts
from helper_funcs import create_env
from helper_funcs import visualize_rendering
from helper_funcs import add_noise
from dynamics_model import Dyn_Model
from mpc_controller import MPCController

from smartstart.reinforcementLearningCore.agents import NavigationRLAgent

def make_save_directories(run_num):
    save_dir = 'run_' + str(run_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/saved_forwardsim')
        os.makedirs(save_dir + '/saved_trajfollow')
        os.makedirs(save_dir + '/training_data')
    return save_dir

def euclidean_distance(state, other_state):
    return np.linalg.norm(state - other_state)

class NND_MB_agent(NavigationRLAgent): #Neural Network Dynamics Model Based Agent (NND_MB_agent)
    tf_datatype = tf.float64
    noiseToSignal = 0.01

    # n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
    actions_ag = 'nc'

    def __init__(self,
                 env,
                 theta=1, # how close to the smart start state you want to navigate to
                 distance_function=euclidean_distance,
                 seed=0,
                 run_num=0,
                 use_existing_training_data=False,  # training data for dynamics model initialization
                 use_existing_dynamics_model=False,
                 desired_traj_type='straight',
                 num_rollouts_save_for_mf=60,
                 might_render=False,
                 visualize_MPC_rollout=False,
                 perform_forwardsim_for_vis=False,
                 print_minimal=False,
                 which_agent=2,
                 use_threading=True,
                 num_rollouts_train=25,
                 num_rollouts_val=20,
                 num_fc_layers=1,
                 depth_fc_layers=500,
                 batchsize=512,
                 lr=0.001,
                 nEpoch=30,
                 fraction_use_new=0.9,
                 horizon=20,
                 num_control_samples=5000,
                 num_aggregation_iters=6,
                 num_trajectories_for_aggregation=10,
                 rollouts_forTraining=9,
                 make_aggregated_dataset_noisy=True,
                 make_training_dataset_noisy=True,
                 noise_actions_during_MPC_rollouts=True,
                 dt_steps=3,
                 steps_per_episode=333,
                 steps_per_rollout_train=333,
                 steps_per_rollout_val=333,
                 min_rew_for_saving=0,
                 visualize_True=True,
                 visualize_False=False):
        """

        :param env:
        :param seed:
        :param run_num:
        :param use_existing_training_data:
        :param use_existing_dynamics_model:
        :param desired_traj_type:
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
        :param lr: learning rate
        :param nEpoch:
        :param fraction_use_new:
        :param horizon:
        :param num_control_samples:
        :param num_aggregation_iters:
        :param num_trajectories_for_aggregation:
        :param rollouts_forTraining:
        :param make_aggregated_dataset_noisy:
        :param make_training_dataset_noisy:
        :param noise_actions_during_MPC_rollouts:
        :param dt_steps:
        :param steps_per_episode:
        :param steps_per_rollout_train:
        :param steps_per_rollout_val:
        :param min_rew_for_saving:
        :param visualize_True:
        :param visualize_False:

        Will Reset Environment if Training Data is needed
        """

        ### Initial Variables ###################################################################
        self.env = env
        self.N = num_control_samples
        self.horizon = horizon
        self.desired_states = None
        self.current_desired_state_index = None #will start at 0 when a new episode begins otherwise error
        self.distances_left = None # self.distances_left[x] will tell the distance from
                                    # desired_states[x] to desired_states[x+1] + [x+1] to [x+2].... until the end
        self.distance_function = distance_function
        self.theta = theta
        if (noise_actions_during_MPC_rollouts):
            self.noise_amount = 0.005
        else:
            self.noise_amount = 0


        save_dir = make_save_directories(run_num) ### make directories for saving data ###
        if seed is not None: # set seeds
            npr.seed(seed)
            tf.set_random_seed(seed)
        self.dt_from_xml = env.env.model.opt.timestep
        random_policy = Policy_Random(env) # create random policy for data collection
        self.sess = tf.Session() #TODO: decide if i need GPU options

        ### Get Training Data  ###########################################################################
        if (use_existing_training_data):
            if (not (print_minimal)):
                print("\n#####################################")
                print("Retrieving training data & policy from saved files")
                print("#####################################\n")

            self.dataX = np.load(save_dir + '/training_data/dataX.npy')  # input1: state
            self.dataY = np.load(save_dir + '/training_data/dataY.npy')  # input2: control
            self.dataZ = np.load(save_dir + '/training_data/dataZ.npy')  # output: nextstate-state
            self.states_val = np.load(save_dir + '/training_data/states_val.npy')
            self.controls_val = np.load(save_dir + '/training_data/controls_val.npy')
            self.forwardsim_x_true = np.load(save_dir + '/training_data/forwardsim_x_true.npy')
            self.forwardsim_y = np.load(save_dir + '/training_data/forwardsim_y.npy')
        else:
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

            #DH - Don't think we need to change the observed state, maybe later
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
            self.forwardsim_x_true, self.forwardsim_y = generate_training_data_inputs(states_forwardsim, controls_forwardsim)

            if (not (print_minimal)):
                print("\n#####################################")
                print("Saving data")
                print("#####################################\n")

            np.save(save_dir + '/training_data/dataX.npy', self.dataX)
            np.save(save_dir + '/training_data/dataY.npy', self.dataY)
            np.save(save_dir + '/training_data/dataZ.npy', self.dataZ)
            np.save(save_dir + '/training_data/states_val.npy', self.states_val)
            np.save(save_dir + '/training_data/controls_val.npy', self.controls_val)
            np.save(save_dir + '/training_data/forwardsim_x_true.npy', self.forwardsim_x_true)
            np.save(save_dir + '/training_data/forwardsim_y.npy', self.forwardsim_y)

        # XYZ variable data ##############################################################################
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

        # Create Dynamics Model and MPC Controller objects ###############################################
        # dimensions
        assert self.inputs.shape[0] == self.outputs.shape[0]
        inputSize = self.inputs.shape[1]
        outputSize = self.outputs.shape[1]

        dataX_new = np.zeros((0, self.dataX.shape[1]))
        dataY_new = np.zeros((0, self.dataY.shape[1]))
        dataZ_new = np.zeros((0, self.dataZ.shape[1]))

        # initialize dynamics model
        self.dyn_model = Dyn_Model(inputSize, outputSize, self.sess, lr, batchsize, num_fc_layers, depth_fc_layers,
                                   self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z,
                                   self.tf_datatype, print_minimal)

        # randomly initialize all vars
        self.sess.run(tf.global_variables_initializer())

        # TODO: ROLLOUTS FOR TRAINING DATA!!!!!

    def get_action(self, state):
        best_action, best_sim_number, best_sequence = self.get_best_sim_actions(state)

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
            action_to_take = action_to_take + noise #no clip maybe go over bounds for action space

        return action_to_take


    def observe(self, state, action, reward, new_state, done):
        # get distances to current and next waypoints

        distance_to_current = self.distance_function(new_state, self.current_desired_state)
        distance_to_next = self.distance_function(new_state, self.next_desired_state)

        # TODO decide which one to use or to use both (theta, or distances to next<=current)
        if distance_to_current <= self.theta or distance_to_next <= distance_to_current:
                # close enough to curr_waypoint, move on to next waypoint
                self.current_desired_state_index = max(self.current_desired_state_index + 1,
                                                       len(self.desired_states) - 1)

    def start_new_episode_plan(self, starting_state, desired_states):
        self.current_desired_state_index = 0
        self.desired_states = desired_states

        #calculate distances beforehand
        if len(desired_states) >= 2:
            self.distances_left = \
                [self.distance_function(desired_states[x-1], desired_states[x]) for x in range(1, len(desired_states))] + [0]
        else:
            self.distances_left = [0]


    @property
    def current_desired_state(self):
        return self.desired_states[self.current_desired_state_index]

    @property
    def next_desired_state(self):
        return self.desired_states[min(self.current_desired_state_index, len(self.desired_states) - 1)]


    def get_best_sim_actions(self, curr_nn_state):
        #randomly sample N candidate action sequences
        all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))

        #forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        resulting_states = self.dyn_model.do_forward_sim([curr_nn_state, 0], np.copy(all_samples), many_in_parallel)
        resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize]

        #init vars to evaluate the trajectories
        scores=np.zeros((self.N,))

        samples_desired_state_indices = np.tile(self.current_desired_state_index, (self.N,))
        samples_desired_state_indices = samples_desired_state_indices.astype(int)


        # moved_to_next = np.zeros((self.N,))
        # done_forever = np.zeros((self.N,))
        # prev_forward = np.zeros((self.N,))
        # prev_pt = resulting_states[0]

        #accumulate reward over each timestep
        for pt_number in range(resulting_states.shape[0]):

            #array of "the point"... for each sim
            pt = resulting_states[pt_number] # N x state

            #get distances to current and next waypoints
            current_desired_states = self.desired_states[samples_desired_state_indices]
            next_desired_states = self.desired_states[np.minimum(samples_desired_state_indices + 1, len(self.desired_states) - 1)]
            distances_to_current = np.apply_along_axis(lambda x: self.distance_function(x, pt), 0, current_desired_states)
            distances_to_next = np.apply_along_axis(lambda x: self.distance_function(x, pt), 0, next_desired_states)

            # boolean array, tells whether or not the sample moves to the next waypoint
            # TODO decide which one to use or to use both( theta, or distances to next<=current)
            move_to_next = np.logical_and(
                np.logical_or(distances_to_current <= self.theta, distances_to_next <= distances_to_current),
                samples_desired_state_indices != len(self.desired_states) - 1)

            # update scores as the sum of distances left on path
            scores[np.logical_not(move_to_next)] += self.distances_left[samples_desired_state_indices] + distances_to_current #TODO calculate distance maybe with reward function?
            scores[move_to_next] += self.distances_left[samples_desired_state_indices + 1] + distances_to_next #TODO calculate distance maybe with reward function?

            # update next waypoint for each sample if necessary
            samples_desired_state_indices[move_to_next] += 1 # if within theta of waypoint, increment waypoint


        #pick best action sequence
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores)
        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])

        return best_action, best_sim_number, best_sequence

