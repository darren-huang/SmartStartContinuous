# imports
import os
import re
import time

import numpy as np
import tensorflow as tf

from smartstart.RLAgents.NND_MB_agent import NND_MB_agent
from smartstart.utilities.datacontainers import Summary, Episode
from smartstart.utilities.numerical import path_deltas_stds_and_means_per_dim, radii_calc
# my imports
# noinspection PyPackageRequirements,PyPackageRequirements
from smartstart.utilities.plot import plot_path, show_plot, ion_plot, pause_plot, update_path, ioff_plot
from smartstart.utilities.utilities import get_default_data_directory, get_start_waypoints_final_states_steps, \
    set_global_seeds

# from rllab.rllab.envs.normalized_env import normalize

if __name__ == "__main__":
    import gym

    # intialize variables###############################################################################
    # directories
    load_training = True
    load_dir = "default"
    save_model = False
    save_dir = "default"

    #network size
    depth_fc_layers = 32
    num_fc_layers = 1
    lr = .001
    nEpochs = 30
    num_episodes_for_aggregation=1

    # waypoint options
    mean_per_stepsize = 1
    std_per_stepsize = 1
    stepsizes_in_waypoint_radii = 1
    steps_per_waypoint = 1

    # MPC reward/controller
    horizon = 4
    gamma = .75
    horizontal_penalty_factor = .5
    N=500

    # trainer options
    num_episodes = 10
    max_steps = 1000

    #display options
    render = False
    render_episode = False
    print_steps = False
    print_results = True
    ENV_NAME = 'MountainCarContinuous-v0'  # configuring environment
    env = gym.make(ENV_NAME)

    RANDOM_SEED = 1234  # Reset the seed for random number generation
    set_global_seeds(RANDOM_SEED)

    with tf.Session() as sess:
        # Initialize agent, see class for available parameters
        agent = NND_MB_agent(env, sess, steps_per_waypoint=steps_per_waypoint, mean_per_stepsize=mean_per_stepsize,
                             std_per_stepsize=std_per_stepsize, stepsizes_in_waypoint_radii=stepsizes_in_waypoint_radii,
                             gamma=gamma, horizontal_penalty_factor=horizontal_penalty_factor,
                             horizon=horizon, num_control_samples=N,
                             num_episodes_for_aggregation=num_episodes_for_aggregation,
                             save_dir_name=save_dir,
                             load_dir_name=load_dir,
                             save_resulting_dynamics_model=save_model,
                             load_existing_training_data=load_training,
                             depth_fc_layers=depth_fc_layers,
                             num_fc_layers=num_fc_layers,
                             lr=lr,
                             nEpoch=nEpochs)  # type: NND_MB_agent

        # intializing the desired_states
        target_default_directory = "0_ddpg_summaries_DEPRACATED"
        # target_file_name = "DDPG_agent_MountainCarContinuous-v0-1000ep.json"
        target_file_name = "DDPG_agent_MountainCarContinuous-v0_test-2ep.json"
        target_file_pathname = os.path.join(get_default_data_directory(target_default_directory), target_file_name)
        target_summary = Summary.load(target_file_pathname)  # type:Summary
        target_path = target_summary.get_last_path(0)
        target_reward = target_summary.get_last_reward(0)
        # target_path = target_summary.best_path
        # target_reward = target_summary.best_reward
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

            # radii calc (based off of standard deviation and mean of step sizes) for the Plotting
            stds, means = path_deltas_stds_and_means_per_dim(target_path)
            radii = radii_calc(means, stds, mean_per_stepsize, std_per_stepsize, stepsizes_in_waypoint_radii)

            # plot settings
            plot = True
            ion_plot()
            axis0, line_collection, line_collection2, line_collection3, highlight = None, None, None, None, None
            plotted = False

            # plot pauser / controller
            pauser = True
            prev_command = "0"
            num_steps_left = 1
            patt = "^(\d+|pause\d+\.?\d*|p\d+\.?\d*)?$"
            for step in range(max_steps):
                # rendering
                if render:
                    render = agent.render(env)

                # user inputs number steps to run
                while num_steps_left <= 0:
                    input_str = input("num steps (+int) >>>")
                    m = re.match(patt, input_str)
                    if m:
                        if not m.group(1):
                            m = re.match(patt, prev_command)
                            input_str = prev_command

                        if m.group(1)[0] == "p" or m.group(1)[0:5] == "pause":
                            pause_plot(float("".join([i for i in m.group(1) if i.isdigit() or i == "."])))
                        else:
                            num_steps_left = int(m.group(1))
                        prev_command = input_str
                num_steps_left -= 1


                # agent action
                # t1 = time.time()
                action, predicted_sequence = agent.get_action_with_predicted_states(observation)
                # print(time.time() - t1)
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
                    axis0, line_collection, line_collection2, line_collection3, \
                    highlight = plot_path(target_path,
                                          path2=episode.get_total_path(),
                                          path3=predicted_sequence,
                                          title="Desired Path rw({0:.2f}) vs. Current Path rw({0:.2f})".format(
                                              target_reward, episode.reward),
                                          x_label="x_pos", y_label="x_velocity",
                                          waypoint_centers=desired_states,
                                          highlight_waypoint_index=agent.current_desired_state_index,
                                          radii=radii)
                    show_plot()
                    plotted = True
                    pause_plot(0.001)
                else:
                    line_collection2, line_collection3, highlight = update_path(axis0, line_collection, line_collection2,
                                                              line_collection3, highlight,
                                                              episode.get_total_path(),
                                                              predicted_sequence,
                                                              agent.current_desired_state, radii)
                    pause_plot(0.001)

                # check terminal observation
                if done:
                    break
                else:
                    observation = new_observation  # increment local observation

            ioff_plot()
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

            ioff_plot()
            show_plot()

        # render results
        while render:
            render = agent.render(env)

        summary.save(get_default_data_directory("nnd_mb_tests"), extra_name_append="")