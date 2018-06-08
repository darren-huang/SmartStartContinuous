import abc


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

    def start_new_episode(self):
        """
        tell the model that a new episode is beginning (really i think only smartStart needs this)
        """
        pass

