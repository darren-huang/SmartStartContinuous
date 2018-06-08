"""SmartStart module

Defines method for generating a SmartStart object from an algorithm object. The
SmartStart object will be a subclass of the original algorithm object.
"""
import numpy as np

from smartstart.RLDiscreteAlgorithms import ValueIteration
from smartstart.utilities.datacontainers import Episode, Summary


def generate_smartstart_object(base, env, *args, **kwargs):
    """Generates SmartStart object

    Algorithms derived from
    :class:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning` can be used to
    construct a SmartStart object. SmartStart becomes a direct subclass from
    the specified base class and inherits all their methods. SmartStart
    objects can be used in the same way as the original base class.

    Note:
        Haven't been able to figure out how to add the SmartStart class to
        the documentation. Because it is defined within this function it is
        hidden in the documentation.

    Parameters
    ----------
    base : :obj:`~smartstart.RLDiscreteAlgorithms.tdlearning.TDLearning`
        base algorithm to convert to SmartStart algorithm
    env : :obj:`~smartstart.environments.environment.Environment`
        environment
    *args :
        see :obj:`~smarstart.smartexploration.smartexploration.SmartStart`
        and the base class for possible parameters
    **kwargs :
        see :obj:`~smarstart.smartexploration.smartexploration.SmartStart`
        and the base class for possible parameters

    Returns
    -------
    :obj:`~smarstart.smartexploration.smartexploration.SmartStart`
        SmartStart algorithm with the specified base class

    """

    class SmartStart(base):
        """SmartStart

        SmartStart algorithm consists of two stages:

            1.  Select smart start
            2.  Guide to smart start

        The smart start is selected using the UCB1 algorithm,
        as described at by the get_start method.

        The guiding to the smart start is done using model-based
        reinforcement learning. A transition model is fit using the
        visitation counts. A reward function that is zero expect for
        transitions that directly transition to the smart start,
        those transitions get a reward of one.

        Subsequently this transition model and the reward function are
        used to find a optimal policy using dynamic programming. This
        implementation uses
        :class:`~smartstart.RLDiscreteAlgorithms.valueiteration.ValueIteration`.

        After reaching the smart start the agent will continue with the
        normal reinforcement learning as described by the base class.

        Parameters
        ----------
        env : :obj:`~smartstart.environments.environment.Environment`
            environment
        exploitation_param : :obj:`float`
            scaling factor for value function in smart start selection
        exploration_param : :obj:`float`
            scaling factor for density in the smart start selection
        eta : :obj:`float` or :obj:~smartstart.utilities.scheduler.Scheduler`
            change of using smart start at the start of an episode or
            executed the normal algorithm
        m : :obj:`int`
            number of state-action visitation counts needed before the
            state-action pair is used in fitting the transition model.
        vi_gamma : :obj:`float`
            discount factor for value iteration
        vi_min_error : :obj:`float`
            minimum error for convergence of value iteration
        vi_max_itr : :obj:`int`
            maximum number of iteration of value iteration
        *args :
            see the base class for possible parameters
        **kwargs :
            see the base class for possible parameters

        Parameters
        ----------
        exploitation_param : :obj:`float`
            scaling factor for value function in smart start selection
        exploration_param : :obj:`float`
            scaling factor for density in the smart start selection
        eta : :obj:`float` or :obj:~smartstart.utilities.scheduler.Scheduler`
            change of using smart start at the start of an episode or
            executed the normal algorithm
        m : :obj:`int`
            number of state-action visitation counts needed before the
            state-action pair is used in fitting the transition model.
        policy: :obj:`~smartstart.RLDiscreteAlgorithms.valueiteration.ValueIteration`
            policy used for guiding to the smart start
        """

        def __init__(self,
                     env,
                     exploitation_param=1.,
                     exploration_param=2.,
                     eta=0.5,
                     m=1,
                     vi_gamma=0.99,
                     vi_min_error=1e-5,
                     vi_max_itr=1000,
                     *args,
                     **kwargs):
            super(SmartStart, self).__init__(env, *args, **kwargs)
            self.__class__.__name__ = "SmartStart_" + base.__name__
            self.exploitation_param = exploitation_param
            self.exploration_param = exploration_param
            self.eta = eta
            self.m = m

            self.policy = ValueIteration(self.env, vi_gamma, vi_min_error,
                                         vi_max_itr)

            # counting the spots that have been chosen as smart_states
            self.smart_state_count = np.zeros((self.env.h, self.env.w), dtype=np.int)

        def get_start(self):
            """Determines the smart start state

            The smart start is determined using the UCB1 algorithm. The UCB1
            algorithm is a well known exploration strategy for multi-arm
            bandit problems. The smart start is chosen according to

            smart_start = \arg\max\limits_s\left(alpha * \max\limits_a Q(s,
            a) + \sqrt{\frac{beta * \log\sum\limits_{s'} C(s'}{C(s}} \right)

            Where
                * \alpha = exploitation_param
                * \beta  = exploration_param

            Returns
            -------
            :obj:`np.ndarray`
                smart start
            """
            count_map = self.get_count_map()
            if count_map is None:
                return None
            possible_starts = np.asarray(np.where(count_map > 0))
            if not possible_starts.any():
                return None

            smart_start = None
            max_ucb = -float('inf')
            for i in range(possible_starts.shape[1]):
                obs = possible_starts[:, i]
                q_values, _ = self.get_q_values(obs)
                q_value = max(q_values)
                ucb = self.exploitation_param * q_value + \
                      np.sqrt((self.exploration_param *
                               np.log(np.sum(count_map)))/count_map[tuple(obs)])
                if ucb > max_ucb:
                    smart_start = obs
                    max_ucb = ucb
            return smart_start

        def get_smart_state_density_map(self):
            """Returns smart state density map for environment

            Density-map will be a numpy array equal to the width (w) and height (h)
            of the environment. Each entry (state) will hold the density
            associated with that smart state.

            Note:
                Only works for 2D-environments that have a w and h attribute.

            Returns
            -------
            :obj:`np.ndarray`
                Density map

            """
            count_map = self.smart_state_count
            total_sum = np.sum(count_map)
            if total_sum == 0:
                return count_map
            return count_map / total_sum

        def train(self, render=False, render_episode=False, print_results=True):
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

                    #update self.smart_state_count
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
                                                     smart_state_density_map=self.get_smart_state_density_map())

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
                                                     smart_state_density_map=self.get_smart_state_density_map())
                if print_results:
                    print(message)
                summary.append(episode)

            while render:
                value_map = self.Q.copy()
                value_map = np.max(value_map, axis=2)
                render = self.env.render(value_map=value_map,
                                         density_map=self.get_density_map(),
                                         smart_state_density_map=self.get_smart_state_density_map())

            return summary

        def dynamic_programming(self, start_state):
            """Fits transition model, reward function and performs dynamic
            programming

            Transition model is fitted using the following equation

                T(s,a,s') = \frac{C(s,a,s'}{C(s,a)}

            Where C(*) is the visitation count. The reward function is zero
            everywhere except for the transition that results in the smart start

                R(s,a,s') = 1 if s' == s_{ss}
                R(s,a,s') = 0 otherwise

            Dynamic programming is done using value iteration.

            Parameters
            ----------
            start_state : :obj:`np.ndarray`
                SmartStart state

            """
            # Reset policy
            self.policy.reset()

            # Fit transition model and reward function
            for obs_c, obs_count in self.count_map.items():
                for action, action_count in obs_count.items():
                    for obs_tp1, count in action_count.items():
                        self.policy.add_obs(obs_c)
                        if obs_tp1 == tuple(start_state):
                            self.policy.R[obs_c + action][obs_tp1] = 1.
                        self.policy.T[obs_c + action][obs_tp1] = \
                            count / sum(self.count_map[obs_c][action].
                                        values())

            # Perform dynamic programming
            self.policy.optimize()

    # Create the actual object and return it
    return SmartStart(env, *args, **kwargs)


if __name__ == "__main__":
    from smartstart.environments.gridworld import GridWorld
    from smartstart.environments.gridworldvisualizer import GridWorldVisualizer
    from smartstart.RLDiscreteAlgorithms import SARSALambda

    directory = '/home/bartkeulen/repositories/smartstart/data/tmp'

    np.random.seed()

    env = GridWorld.generate(GridWorld.EASY)
    visualizer = GridWorldVisualizer(env)
    visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                              GridWorldVisualizer.CONSOLE,
                              GridWorldVisualizer.VALUE_FUNCTION,
                              GridWorldVisualizer.DENSITY)
    env.visualizer = visualizer
    # env.T_prob = 0.1

    # env.wall_reset = True

    agent = generate_smartstart_object(SARSALambda, env, eta=0.75, alpha=0.3,
                                       num_episodes=1000, max_steps=1000,
                                       exploitation_param=0.)

    summary = agent.train(render=False, render_episode=True)

    summary.save(directory=directory)
