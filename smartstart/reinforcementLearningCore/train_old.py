import numpy as np

from utilities.datacontainers import Summary, Episode


def train(agent, env, render=False, render_episode=False, print_results=True):
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
    :class:`~smartexploration.utilities.datacontainers.Summary`
        Summary Object containing the training data

    """
    summary = Summary(agent.__class__.__name__ + "_" + env.name)

    for i_episode in range(agent.num_episodes):
        episode = Episode()

        state = env.reset()
        action = agent.get_action(state)

        for _ in range(agent.max_steps):
            state, action, done, render = agent.take_step(state, action, episode,
                                                       render)

            if done:
                break

        # Render and/or print results
        message = "Episode: %d, steps: %d, reward: %.2f" % (
            i_episode, len(episode), episode.average_reward())
        if render or render_episode:
            value_map = agent.Q.copy()
            value_map = np.max(value_map, axis=2)
            render_episode = env.render(value_map=value_map,
                                             density_map=agent.get_density_map(),
                                             message=message)

        if print_results:
            print("Episode: %d, steps: %d, reward: %.2f" % (
                i_episode, len(episode), episode.average_reward()))
        summary.append(episode)

    while render:
        value_map = agent.Q.copy()
        value_map = np.max(value_map, axis=2)
        render = env.render(value_map=value_map,
                                 density_map=agent.get_density_map())

    return summary

def take_step(agent, env, obs, action, episode, render=False, smart_state_density_map=None):
    """Takes a step and updates

    Action action is executed and response is observed. Response is then
    used to update the value function. Data is stored in Episode object.

    Parameters
    ----------
    obs : :obj:`list` of :obj:`int` or :obj:`np.ndarray`
        observation
    action : :obj:`int`
        action
    episode : :obj:`Episode`
        Data container for all the episode data
    render : :obj:`bool`
        True when rendering every time-step (Default value = False)

    Returns
    -------
    :obj:`list` of :obj:`int` or `np.ndarray`
        next observation
    :obj:`int`
        next action
    :obj:`bool`
        done, True when obs_tp1 is terminal state
    :obj:`bool`
        render, True when rendering must continue
    """
    obs_tp1, reward, done, _ = env.step(action)

    if render:
        value_map = agent.Q.copy()
        value_map = np.max(value_map, axis=2)
        render = env.render(value_map=value_map,
                            density_map=agent.get_density_map(),
                            smart_start_state_density_map=smart_state_density_map)

    _, action_tp1 = agent.update_q_value(obs, action, reward, obs_tp1, done)

    agent.increment(obs, action, obs_tp1)

    episode.append(obs, action, reward, obs_tp1, done)

    return obs_tp1, action_tp1, done, render


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

def train_smart_start(self, render=False, render_episode=False, print_results=True):
    """Runs a training experiment

    Training experiment runs for self.num_episodes and each episode
    takes a maximum of self.max_steps.

    At the start of each episode there is a eta change of using smart
    start. When smart start is not used the agent will use normal
    reinforcement learning as described by the base class.

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
    :class:`~smartexploration.utilities.datacontainers.Summary`
        Summary Object containing the training data

    """
    summary = Summary(self.__class__.__name__ + "_" + self.env.name)

    for i_episode in range(self.num_episodes):
        episode = Episode()

        obs = self.env.reset()

        max_steps = self.max_steps

        # eta probability of using smart start
        if i_episode > 0 and np.random.rand() <= self.eta:
            # Step 1: Choose smart start
            start_state = self.get_start()

            # update self.smart_state_count
            self.smart_state_count[start_state[0]][start_state[1]] += 1

            # Step 2: Guide to smart start
            self.dynamic_programming(start_state)

            finished = False
            for i in range(self.max_steps):
                action = self.policy.get_action(obs)
                obs, _, done, render = self.take_step(obs, action,
                                                      episode, render,
                                                      smart_state_density_map=self.get_smart_state_density_map())

                if np.array_equal(obs, start_state):
                    break

                if done:
                    finished = True
                    break

            max_steps = self.max_steps - len(episode)

            if finished:
                max_steps = 0

        if render or render_episode:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render_episode = self.env.render(value_map=value_map,
                                             density_map=self.get_density_map(),
                                             smart_start_state_density_map=self.get_smart_state_density_map())

        # Perform normal reinforcement learning
        action = self.get_action(obs)
        for step in range(max_steps):
            obs, action, done, render = self.take_step(obs, action,
                                                       episode, render,
                                                       smart_state_density_map=self.get_smart_state_density_map())

            if done:
                break

        # Render and/or print results
        message = "Episode: %d, steps: %d, reward: %.2f" % \
                  (i_episode, len(episode), episode.total_reward())
        if render or render_episode:
            value_map = self.Q.copy()
            value_map = np.max(value_map, axis=2)
            render_episode = self.env.render(value_map=value_map,
                                             density_map=self.get_density_map(),
                                             message=message,
                                             smart_start_state_density_map=self.get_smart_state_density_map())
        if print_results:
            print(message)
        summary.append(episode)

    while render:
        value_map = self.Q.copy()
        value_map = np.max(value_map, axis=2)
        render = self.env.render(value_map=value_map,
                                 density_map=self.get_density_map(),
                                 smart_start_state_density_map=self.get_smart_state_density_map())

    return summary