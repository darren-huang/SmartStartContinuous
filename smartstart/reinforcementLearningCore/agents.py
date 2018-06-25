import abc
from reinforcementLearningCore.counter import Counter

class RLAgent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, state):
        """
        given a state, calculates the action to take in the environment

        :param state: the current state to take an action from
        :return: action appropriate for the environment
        """

    @abc.abstractmethod
    def observe(self, state, action, reward, new_state, done):
        """
        after taking an action in the environment, observe the new state and reward (updates models)

        :param state: the state before the action
        :param action:  the action take
        :param reward:  the resulting reward for the (state,action) pair
        :param new_state:  the resulting state for the (state,action) pari
        :param done:  whether or not the action resulted in a terminal action
        """

    @abc.abstractmethod
    def render(self, env, **kwargs):
        """
        given the environment it is already in, renders the environment

        :param env: the environment the agent is currently in
        """

    def start_new_episode(self, state):
        """
        tell the model that a new episode is beginning (really i think only smartStart needs this)
        """
        pass

    def end_episode(self):
        """
        tell the model that the current episode has ended (ie. terminal state or max steps reached)
        """
        pass

class NavigationRLAgent(RLAgent):
    """
    RL agent that at the beginning of an episode decides to plan it's trajectory
    """
    def start_new_episode_plan(self, state, desired_sates):
        """
        :param state: starting state for the episode
        :param desired_sates: the desired states it should try to navigate to in order
        """


class ValueFuncRLAgent(RLAgent):
    """
        RLAgent who has a value function
        """

    @abc.abstractmethod
    def get_state_value(self, state):
        """
        given the agent's state, the agent returns its project value for that state (max of all q-values for that state)

        :param state: observes state inside agent's environment
        :return: value of that state (usually a float)
        """

class ValueFuncAndCountRLAgent(ValueFuncRLAgent, Counter):
    """
    RLAgent who also is a Counter (see smartstart.RLDiscreteAlgorithms.counter.Counter)
    """