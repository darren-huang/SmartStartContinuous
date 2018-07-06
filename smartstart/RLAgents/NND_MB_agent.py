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
from smartstart.utilities.plot import plot_path, show_plot, ion_plot, pause_plot, update_path
from smartstart.utilities.datacontainers import Summary, Episode
from smartstart.utilities.utilities import get_default_directory, get_start_waypoints_final_states_steps
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim


# from rllab.rllab.envs.normalized_env import normalize


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
    """

    :param state: state/list of states (numpy) EITHER 1D or 2D
    :param other_state: state/list of states (numpy) EITHER 1D or 2D
    :return: distance/ list of distances (numpy)
    """
    axis = max(len(state.shape), len(other_state.shape)) - 1
    return np.sum((state - other_state) ** 2, axis=axis) ** 0.5


def elliptical_euclidean_distance_function_generator(radii):
    """
      __
     / | \                    |
    |  | | where the vertical | represents the y-radius
    |  ._| and the horizontal _ represents the x-radius
    |    | (distance function makes all points on that ellipse become distance 1 from the center
     \__/
    :param radii: for states of n-dimensions, radii is n - long specifying the elliptical radius along that axis
    :return:
    """
    for i in radii:
        assert i > 0
    radii = np.asarray(radii)

    def distance_func(state, other_state):
        """

        :param state: state/list of states (numpy) EITHER 1D or 2D
        :param other_state: state/list of states (numpy) EITHER 1D or 2D
        :return: distance/ list of distances (numpy)
        """
        axis = max(len(state.shape), len(other_state.shape)) - 1
        return np.sum(((state - other_state) / radii) ** 2, axis=axis) ** 0.5

    return distance_func


class NND_MB_agent(NavigationRLAgent):  # Neural Network Dynamics Model Based Agent (NND_MB_agent)
    tf_datatype = tf.float64
    noiseToSignal = 0.01

    # n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
    actions_ag = 'nc'

    def __init__(self, env, replay_buffer=None, BUFFER_SIZE=10000, theta=1,
                 steps_per_waypoint=10, mean_per_stepsize=1, std_per_stepsize=1, stepsizes_in_waypoint_radii=1,
                 seed=0, run_num=0,
                 use_existing_training_data=False, use_existing_dynamics_model=False, num_rollouts_save_for_mf=60,
                 print_minimal=False, which_agent=2, use_threading=True, num_rollouts_train=25, num_rollouts_val=20,
                 num_fc_layers=1, depth_fc_layers=500, batchsize=512, lr=0.001, nEpoch=30, fraction_use_new=0.9,
                 horizon=20, num_control_samples=5000, num_episodes_for_aggregation=10, rollouts_forTraining=9,
                 make_aggregated_dataset_noisy=True, make_training_dataset_noisy=True,
                 noise_actions_during_MPC_rollouts=True, dt_steps=3, steps_per_rollout_train=333,
                 steps_per_rollout_val=333, visualize_False=False):
        """
        :param env: the environment the agent is going to navigate
        :param replay_buffer: the buffer that stores previous experiences (state, action, reward, terminal, next_state) tuples
        :param BUFFER_SIZE: max size of the memory buffer
        :param theta: the minimum distance for the agent to consider a waypoint "reached" - distance defined by distance function
        :param seed: for the random packages
        :param run_num: the number that labels the run, determines the name of the folder to save/load to/from
        :param use_existing_training_data: whether or not to load the training data
        :param use_existing_dynamics_model: whether or not to load the dynamics model
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

        ### Initial Variables ###################################################################
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

        if seed is not None:  # set seeds
            npr.seed(seed)
            tf.set_random_seed(seed)

        self.sess = tf.Session()  # TODO: decide if i need GPU options

        # Initialize replay memory
        if replay_buffer == None:
            self.replay_buffer = ReplayBuffer(self, BUFFER_SIZE)  # type: ReplayBuffer
        else:
            self.replay_buffer = replay_buffer  # type: ReplayBuffer

        ### Get Training Data  ###########################################################################
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

        # Create Dynamics Model ###############################################
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
            action_to_take = action_to_take + noise  # no clip maybe go over bounds for action space

        return action_to_take

    def observe(self, state, action, reward, new_state, done):
        # get distances to current and next waypoints

        distance_to_current = self.distance_function(new_state, self.current_desired_state)
        distance_to_next = self.distance_function(new_state, self.next_desired_state)

        # TODO decide which one to use or to use both (theta, or distances to next<=current)
        if distance_to_current <= self.theta or distance_to_next <= distance_to_current:
            # close enough to curr_waypoint, move on to next waypoint
            self.current_desired_state_index = min(self.current_desired_state_index + 1,
                                                   len(self.desired_states) - 1)

    def start_new_episode_plan(self, starting_state, path_to_follow):
        """
        Called at the beginning of an episode.
        :param starting_state: the starting state of the episode
        :param desired_states: the states it wants to navigate through, agent will try to follow
            route and end up within close proximity to the last state
        """
        # set variables for the new episode
        self.current_desired_state_index = 0
        self.path_to_follow = path_to_follow
        desired_states = np.asarray(get_start_waypoints_final_states_steps(path_to_follow, self.steps_per_waypoint))
        self.desired_states = desired_states
        stds, means = path_deltas_stds_and_means_per_dim(path_to_follow)
        stepsizes = (means * self.mean_per_stepsize) + (stds * self.std_per_stepsize)
        radii = stepsizes * self.stepsizes_in_waypoint_radii
        self.distance_function = \
            elliptical_euclidean_distance_function_generator(radii)


        # calculate distances for the MPC reward function
        if len(desired_states) >= 2:
            self.distances_left = \
                [self.distance_function(desired_states[x - 1], desired_states[x]) for x in
                 range(1, len(desired_states))] + [0]
        else:
            self.distances_left = [0]
        self.distances_left = np.asarray(self.distances_left)

        # train
        if self.num_episodes_finished % self.num_episodes_for_aggregation == 0:
            self.train_dynamics_model()
        self.num_episodes_finished += 1

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
        if self.make_aggregated_dataset_noisy:
            s_batch = add_noise(s_batch)
            s2_batch = add_noise(s2_batch)
        dataX_new_preprocessed = np.nan_to_num((s_batch - self.mean_x) / self.std_x)
        dataY_new_preprocessed = np.nan_to_num((a_batch - self.mean_y) / self.std_y)
        dataZ_new_preprocessed = np.nan_to_num((s2_batch - self.mean_z) / self.std_z)

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
        return self.desired_states[min(self.current_desired_state_index, len(self.desired_states) - 1)]

    def get_best_sim_actions(self, curr_nn_state):
        # randomly sample N candidate action sequences
        all_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high,
                                  (self.N, self.horizon, self.env.action_space.shape[0]))

        # forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        resulting_states = self.dyn_model.do_forward_sim([curr_nn_state, 0], np.copy(all_samples), many_in_parallel)
        resulting_states = np.array(resulting_states)  # this is [horizon+1, N, statesize]

        # init vars to evaluate the trajectories
        scores = np.zeros((self.N,))

        samples_desired_state_indices = np.tile(self.current_desired_state_index, (self.N,))
        samples_desired_state_indices = samples_desired_state_indices.astype(int)

        # moved_to_next = np.zeros((self.N,))
        # done_forever = np.zeros((self.N,))
        # prev_forward = np.zeros((self.N,))
        # prev_pt = resulting_states[0]

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
            # TODO decide which one to use or to use both( theta, or distances to next<=current)
            move_to_next = np.logical_and(
                np.logical_or(distances_to_current <= self.theta, distances_to_next <= distances_to_current),
                samples_desired_state_indices != len(self.desired_states) - 1)

            # update scores as the sum of distances left on path
            not_move_to_next = np.logical_not(move_to_next)
            scores[not_move_to_next] += \
                self.distances_left[samples_desired_state_indices[not_move_to_next]] + \
                distances_to_current[not_move_to_next]  # TODO calculate distance maybe with reward function?
            scores[move_to_next] += \
                self.distances_left[samples_desired_state_indices[move_to_next] + 1] + \
                distances_to_next[move_to_next]  # TODO calculate distance maybe with reward function?

            # update next waypoint for each sample if necessary
            samples_desired_state_indices[move_to_next] += 1  # if within theta of waypoint, increment waypoint

        # pick best action sequence
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores)
        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])

        return best_action, best_sim_number, best_sequence


if __name__ == "__main__":
    import random
    import gym

    # intialize variables
    steps_per_waypoint = 10
    num_episodes = 2
    max_steps = 1000
    render = False
    render_episode = False
    print_steps = True
    print_results = True
    ENV_NAME = 'MountainCarContinuous-v0'  # configuring environment
    env = gym.make(ENV_NAME)
    RANDOM_SEED = 1234  # Reset the seed for random number generation
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # Initialize agent, see class for available parameters
    agent = NND_MB_agent(env,
                         theta=1,
                         steps_per_waypoint=steps_per_waypoint,
                         mean_per_stepsize= 1,
                         std_per_stepsize= 1,
                         stepsizes_in_waypoint_radii=1,
                         use_existing_training_data=True,
                         horizon=20,
                         num_episodes_for_aggregation=1)  # type: NND_MB_agent

    # intializing the desired_states
    target_default_directory = "ddpg_summaries"
    target_file_name = "DDPG_agent_MountainCarContinuous-v0-1000ep.json"
    target_file_pathname = os.path.join(get_default_directory(target_default_directory), target_file_name)
    target_summary = Summary.load(target_file_pathname)  # type:Summary
    target_path = target_summary.get_last_path(0)
    target_reward = target_summary.get_last_reward(0)
    desired_states = get_start_waypoints_final_states_steps(target_path, steps_per_waypoint)

    # summary object
    summary = Summary(agent.__class__.__name__ + "_" + env.spec.id)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})  # for printing

    # begin training episodes(1)
    for i_episode in range(num_episodes):
        episode = Episode()  # type: Episode # record the episode
        observation = env.reset()  # setup env

        # Step through the Episode
        agent.start_new_episode_plan(observation, target_path)  # only needed for smartStart

        plot_pauser_count = 0
        prev_pause_count = 1
        plot = True
        ion_plot()
        axis0, line_collection, line_collection2, highlight = None, None, None, None
        plotted = False
        for step in range(max_steps):

            # rendering
            if render:
                render = agent.render(env)

            # agent action
            action = agent.get_action(observation)

            # environment processing
            new_observation, reward, done, _ = env.step(action)  # also returns emtpy dict (to match openAI)

            # printing the step
            if print_steps:
                print("        Step: {}, State: {}, Action: {}, New_State: {}, Reward: {}".format(step, observation,
                                                                                                  action,
                                                                                                  new_observation,
                                                                                                  reward).replace("\n",
                                                                                                                  ""))

            # agent update model// observe new observation
            agent.observe(observation, action, reward, new_observation, done)

            # record results to episode object
            episode.append(observation, action, reward, new_observation, done)

            if not plotted:
                axis0, line_collection, line_collection2, \
                highlight = plot_path(target_path,
                                      path2=episode.get_total_path(),
                                      title="Desired Path rw({0:.2f}) vs. Current Path rw({0:.2f})".format(
                                          target_reward, episode.reward),
                                      x_label="x_pos", y_label="x_velocity",
                                      waypoint_centers=desired_states,
                                      highlight_waypoint_index=agent.current_desired_state_index,
                                      radii=[1, 1])
                show_plot()
                plotted = True
                pause_plot(0.001)
            else:
                line_collection2, highlight = update_path(axis0, line_collection, line_collection2, highlight,
                                                          episode.get_total_path(),
                                                          agent.current_desired_state, [1, 1])
                pause_plot(0.001)

            # check terminal observation
            if done:
                break
            else:
                observation = new_observation  # increment local observation
        agent.end_episode()  # needed for continuous stuffs
        # Episode over

        # Final Render and/or print results
        if render or render_episode:
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward())
            render_episode = agent.render(env, message=message)
        if print_results:
            print("Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward()))

        # update summary
        summary.append(episode)

    # render results
    while render:
        render = agent.render(env)

    summary.save(get_default_directory("nnd_mb_tests"), extra_name_append="")
