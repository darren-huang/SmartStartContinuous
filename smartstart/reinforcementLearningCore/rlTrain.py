from smartstart.utilities.datacontainers import Summary, Episode
import numpy as np



def rlTrain(agent, env, render=False, render_episode=False, print_results=True, print_steps=True,
            num_episodes=500,
            max_steps=1000):
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
    #summary object
    summary = Summary(agent.__class__.__name__ + "_" + env.spec.id)

    # for printing
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    #begin training episodes(1)
    for i_episode in range(num_episodes):
        episode = Episode() #record the episode
        observation = env.reset() #setup env

        # Step through the Episode
        agent.start_new_episode(observation) # only needed for smartStart
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
                print("        Step: {}, State: {}, Action: {}, New_State: {}, Reward: {}".format(step, observation, action, new_observation, reward).replace("\n", ""))

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
            print("Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward()))

        #update summary
        summary.append(episode)

    #render results
    while render:
        render = agent.render(env)

    return summary



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