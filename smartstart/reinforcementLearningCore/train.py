from utilities.datacontainers import Summary, Episode
import numpy as np

def train(agent, env, render=False, render_episode=False, print_results=True):
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
    summary = Summary(agent.__class__.__name__ + "_" + env.name)

    #begin training episodes(1)
    for i_episode in range(agent.num_episodes):
        episode = Episode() #record the episode
        state = env.reset() #setup env

        # Step through the Episode
        agent.start_new_episode()
        for _ in range(agent.max_steps):
            #rendering
            if render:
                render = agent.render(env)

            #agent action
            action = agent.get_action(state)

            #environment processing
            new_state, reward, done, _ = env.step(action) # also returns emtpy dict (to match openAI)

            #agent update model// observe new state
            agent.observe(state, action, reward, new_state, done)

            #record results to episode object
            episode.append(state, action, reward, new_state, done)

            #check terminal state
            if done:
                break
        # Episode over

        # Final Render and/or print results
        if render or render_episode:
            message = "Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward())
            render = agent.render(env, message=message)
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