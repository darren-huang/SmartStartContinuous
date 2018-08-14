import re

import numpy as np

import time
from smartstart.smartexploration.smartexplorationcontinuous import SmartStartContinuous
from smartstart.utilities.datacontainers import Summary, Episode
from utilities.numerical import path_deltas_stds_and_means_per_dim
from utilities.plot import ion_plot, pause_plot, plot_path, show_plot, update_path, ioff_plot
from utilities.utilities import get_default_data_directory


def rlTrain(agent, env, render=False, render_episode=False, print_results=True, print_steps=True,
            num_episodes=500,
            max_steps=1000,
            progress_bar=False,
            id=None,
            num_ticks=100,
            print_time=False):
    """Runs a training experiment

    Training experiment runs for agent.num_episodes and each episode takes
    a maximum of agent.max_steps.

    Parameters
    ----------
    agent : :obj:'~smartexploration.reinforcementLearningCore.agents.RLAgent'
        the reinforcement learning agent to use
    env : :obj:'~smartexploration.environments.environment.Environment'
        the environment the agent is supposed to learn
    render : :obj:`bool`
        True when rendering every time-step (Default value = False)
    render_episode : :obj:`bool`
        True when rendering every episode (Default value = False)
    print_results : :obj:`bool`
        True when printing results to console (Default value = True)

    Returns
    -------
    :class:`~smartexploration.utilities.datacontainers.Summary`
        Summary Object containing the training data

    """
    if progress_bar:
        progress_tick = num_episodes // num_ticks
        print("process id: " + str(id) + "|" + ("-" * num_ticks) + "|    " +
              str(0) + "/" + str(num_episodes) + " episodes")

    #summary object
    #TODO fix how the names are gotten (hasattr func)
    if hasattr(agent, 'get_summary_name'):
        summary = Summary(agent.get_summary_name() + "_" + env.spec.id)
    else:
        summary = Summary(agent.__class__.__name__ + "_" + env.spec.id)
    summary.set_agent(agent)

    # for printing
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    times = [time.time()]

    #begin training episodes(1)
    for i_episode in range(num_episodes):
        if print_time:
            print("\nEpisode " + str(i_episode) + " begins!")

        episode = Episode() #record the episode
        observation = env.reset() #setup env

        # Step through the Episode
        agent.start_new_episode(observation) # only needed for smartStart
        if isinstance(agent, SmartStartContinuous) and agent.smart_start_pathing:
            summary.start_smart_start_episode()

        for step in range(max_steps):
            #rendering
            if render:
                render = agent.render(env)

            #agent action
            action = agent.get_action(observation)

            #environment processing
            new_observation, reward, done, _ = env.step(action) # also returns emtpy dict (to match openAI)

            #printing the step
            if print_steps:
                print("\n" + "        Step: {}, State: {}, Action: {}, New_State: {}, Reward: {}  ||  ".format(step, observation, action, new_observation, reward).replace("\n", ""), end='')

            #agent update model// observe new observation
            agent.observe(observation, action, reward, new_observation, done)

            #record results to episode object
            episode.append(observation, action, reward, new_observation, done)

            #check terminal observation
            if done:
                break
            else:
                observation = new_observation #increment local observation
        agent.end_episode() # needed for continuous stuffs
        # Episode over

        # Final Render and/or print results
        if render or render_episode:
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward())
            render_episode = agent.render(env, message=message)
        if print_results:
            print("\n\nEpisode: %d, steps: %d, reward: %.2f || " % (
                i_episode, len(episode), episode.average_reward()) , end='')

        #update summary
        summary.append(episode)

        if progress_bar and (i_episode + 1) % progress_tick == 0:
            i_progress = (i_episode + 1) // progress_tick
            i_rest = num_ticks - i_progress
            print("process id: " + str(id) + "|" +  ("*"*i_progress) + ("-"*i_rest) + "|    " +
                  str(i_episode + 1) + "/" + str(num_episodes) + " episodes")

        if print_time:
            times.append(time.time())
            if len(times) >= 2:
                print("took: " + str(times[-1] - times[-2]))
            print("end")


    #render results
    while render:
        render = agent.render(env)

    return summary


def rlTrainGraphSS(agent, env,
                   render=False, render_episode=False, print_results=True, print_steps=True,
                   num_episodes=500,
                   max_steps=1000,
                   plot_ss_stuff=True,
                   print_time=False):
    """

    :type agent: SmartStartContinuous
    """
    # summary object
    if hasattr(agent, 'get_summary_name'):
        summary = Summary(agent.get_summary_name() + "_" + env.spec.id)
    else:
        summary = Summary(agent.__class__.__name__ + "_" + env.spec.id)
    summary.set_agent(agent)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})  # for printing

    times = [time.time()]

    # begin training episodes(1)
    for i_episode in range(num_episodes):
        if print_time:
            print("\nEpisode " + str(i_episode) + " begins!")

        episode = Episode()  # type: Episode # record the episode
        observation = env.reset()  # setup env

        # Step through the Episode
        agent.start_new_episode(observation)  # only needed for smartStart
        if isinstance(agent, SmartStartContinuous) and agent.smart_start_pathing:
            summary.start_smart_start_episode()

        plot_this_episode = agent.smart_start_pathing and plot_ss_stuff

        # radii calc (based off of standard deviation and mean of step sizes) for the Plotting
        radii = agent.nnd_mb_agent.radii

        # plot settings
        ion_plot()
        axis0, line_collection, line_collection2, line_collection3, highlight = None, None, None, None, None
        plotted = False

        # plot pauser / controller
        prev_command = "0"
        num_steps_left = 1
        patt = "^(\d+|pause\d+\.?\d*|p\d+\.?\d*)?$"
        for step in range(max_steps):
            # rendering
            if render:
                render = agent.render(env)

            # user inputs number steps to run
            while num_steps_left <= 0 and plot_this_episode:
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
            if agent.smart_start_pathing:
                action, predicted_sequence = agent.nnd_mb_agent.get_action_with_predicted_states(observation)
            else:
                action = agent.get_action(observation)

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

            if plot_this_episode:
                if not plotted:
                    axis0, line_collection, line_collection2, line_collection3, \
                    highlight = plot_path(agent.nnd_mb_agent.path_to_follow,
                                          path2=episode.get_total_path(),
                                          path3=predicted_sequence,
                                          title="Desired Path vs. Current Path",
                                          x_label="x_pos", y_label="x_velocity",
                                          waypoint_centers=agent.nnd_mb_agent.desired_states,
                                          highlight_waypoint_index=agent.nnd_mb_agent.current_desired_state_index,
                                          radii=radii)
                    plot_path(agent.smart_start_path)
                    show_plot()
                    plotted = True
                    pause_plot(0.001)
                else:
                    line_collection2, line_collection3, highlight = update_path(axis0, line_collection, line_collection2,
                                                              line_collection3, highlight,
                                                              episode.get_total_path(),
                                                              predicted_sequence,
                                                              agent.nnd_mb_agent.current_desired_state, radii)
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

        if print_time:
            times.append(time.time())
            if len(times) >= 2:
                print("took: " + str(times[-1] - times[-2]))
            print("end")

        if plot_this_episode:
            ioff_plot()
            show_plot()


    # render results
    while render:
        render = agent.render(env)

    summary.save(get_default_data_directory("nnd_mb_tests"), extra_name_append="")



def trainTDLearningLambda(agent, env, render=False, render_episode=False, print_results=True):
    """Runs a training experiment

    Training experiment runs for agent.num_episodes and each episode takes
    a maximum of agent.max_steps.

    Parameters
    ----------
    render : :obj:`bool`
        True when rendering every time-step (Default value = False)
    render_episode : :obj:`bool`
        True when rendering every episode (Default value = False)
    print_results : :obj:`bool`
        True when printing results to console (Default value = True)

    Returns
    -------
    :obj:`Summary`
        Summary Object containing the training data

    """
    summary = Summary(agent.__class__.__name__ + "_" + env.name)

    for i_episode in range(agent.num_episodes):
        episode = Episode()

        obs = env.reset()
        action = agent.get_action(obs)

        for _ in range(agent.max_steps):
            obs, action, done, render = agent.take_step(obs, action, episode,
                                                       render)

            if done:
                break

        # Clear traces after episode
        agent.traces = np.zeros(
            (env.w, env.h, env.num_actions))

        # Render and/or print results
        message = "Episode: %d, steps: %d, reward: %.2f" % (
            i_episode, len(episode), episode.total_reward())
        if render or render_episode:
            value_map = agent.Q.copy()
            value_map = np.max(value_map, axis=2)
            render_episode = env.render(value_map=value_map,
                                             density_map=agent.get_density_map(),
                                             message=message)
        if print_results:
            print("Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.total_reward()))
        summary.append(episode)

    while render:
        value_map = agent.Q.copy()
        value_map = np.max(value_map, axis=2)
        render = env.render(value_map=value_map,
                                 density_map=agent.get_density_map())

    return summary