""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
import pickle
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    """
    A buffer with a First in First Out policy,
    Stores vectors of (state, action, reward, terminal, new_state),
    Upon calling the start_new_episode method creates a marker to mark where said episode begins
    Can create sample batches (random entries from the buffer)
    NOTE: the add method only allows the self.main_agent from adding entries, this way if multiple agents try to add
            to the replay buffer, only one state entry will add (this is the case with Smart Starts)
    """

    def __init__(self, main_agent, max_buffer_size):
        """
        The right side of the deque contains the most recent experiences
        """
        self.main_agent = main_agent #only the main agent can add observations

        self.max_buffer_size = max_buffer_size

        # indexing number used to distinguish which episode we are on
        # ie.   buffer of size 3:      0 1 2 [3 4 5]
        #       episode numbers           -2 -1 0 1
        #                                  E    E E
        # 1. where the E's show where the episodes begin
        # 2. the episode numbers provide a stable reference for where the episodes begin/end
        #           this is necessary because the indexing of the buffer keeps changing
        #               use self.episode_number_to_buffer_index() to convert from episode number to buffer index
        self.next_episode_number = 0
        self.episode_starting_indices = deque() # holds the COUNT of the first state of the episode
        self.buffer = deque()

    def set_main_agent(self, new_main_agent):
        self.main_agent = new_main_agent

    def add(self, observing_agent, s, a, r, t, s2):
        if observing_agent is not self.main_agent:
            return #main agent should do the observation, all other agents ignored

        experience = (s, a, r, t, s2)
        if len(self.buffer) < self.max_buffer_size:
            self.buffer.append(experience)
        else:
            if len(self.episode_starting_indices) > 0 and \
                    self.episode_starting_indices[0] == self.next_episode_number - self.max_buffer_size:
                # if an episode index is beyond the range of the buffer, remove it
                self.episode_starting_indices.popleft()

                # Decrease rest of episode index numbers and count so first/next episode is 0 (or close to 0)
                if len(self.episode_starting_indices) > 0:
                    first_index = self.episode_starting_indices[0]
                    for i in range(len(self.episode_starting_indices)):
                        self.episode_starting_indices[i] -= first_index
                    self.next_episode_number -= first_index
                else:
                    self.next_episode_number = 0 #count reset to 0 to prevent super large numbers

            self.buffer.popleft()
            self.buffer.append(experience)

        self.next_episode_number += 1

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(list(self.buffer), batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def all_batch(self):
        s_batch = np.array([_[0] for _ in self.buffer])
        a_batch = np.array([_[1] for _ in self.buffer])
        r_batch = np.array([_[2] for _ in self.buffer])
        t_batch = np.array([_[3] for _ in self.buffer])
        s2_batch = np.array([_[4] for _ in self.buffer])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def get_all_states(self):
        return np.array([_[0] for _ in self.buffer] + [self.buffer[-1][4]])

    def clear(self):
        self.buffer.clear()
        self.next_episode_number = 0

    def start_new_episode(self, observing_agent):
        if observing_agent is not self.main_agent:
            return #main agent should do the episode starts, all other agents ignored
        if len(self.episode_starting_indices) > 0 and self.episode_starting_indices[-1] == self.next_episode_number:
            print("shouldn't call start episode twice in a row")
        else:
            self.episode_starting_indices.append(self.next_episode_number)

    def save(self):
        print('saving the replay buffer')
        print('.')
        file = open('replay_buffer.obj', 'wb')
        print('..')
        pickle.dump(self.buffer, file)
        print('...')
        print('the replay buffer was saved succesfully')

    def load(self):
          
        try:
            filehandler = open('replay_buffer.obj', 'rb') 
            self.buffer = pickle.load(filehandler)
            self.next_episode_number = len(self.buffer)
            print('the replay buffer was loaded succesfully')
        except: 
            print('there was no file to load')

    def get_possible_smart_start_indices(self, n_ss):
        """
        Gets possible smart start states indices corresponding to the buffer indices
        note smart start state refers to the s2 variable of the step corresponding to the buffer index
        NOTE: never picks a starting state, partially for convenience, also since it should be the case that the starting
        states would be visited more
        :param n_ss: the number of smart starts
        :return: array of possible smart starts
        """
        if len(self.episode_starting_indices) == 0:
            return []
        #mainly has to ensure the whole episode containing the smartstart state is inside the buffer
        first_buffer_index = self.episode_number_to_buffer_index(self.episode_starting_indices[0])
        number_of_states = min(n_ss, len(self.buffer) - first_buffer_index) # max possible states to

        # sample WITHOUT replacement
        return random.sample(range(first_buffer_index, len(self.buffer)), number_of_states)

    def get_episodic_path_to_buffer_index(self, buffer_index):
        """
        Search for the stored path taken to reach the buffer index (looks for the beginning of the corresponding
        episode then returns the path to the buffer_index)
        :param buffer_index: the index you want the path to
        :return: array of States
        """
        if len(self.episode_starting_indices) == 0:
            raise ValueError(": (   -   no episodes have been recorded")
        episode_index = self.buffer_index_to_episode_number(buffer_index)
        episode_start_index = None

        # shouldn't be a bottle neck because should be a small number of episodes (if it is a bottleneck can use binary search)
        for i in range(len(self.episode_starting_indices) - 1):
            if self.episode_starting_indices[i] <= episode_index and episode_index < self.episode_starting_indices[i+1]:
                episode_start_index = self.episode_starting_indices[i]
                break
        if episode_start_index is None and self.episode_starting_indices[-1] <= episode_index:
            episode_start_index = self.episode_starting_indices[-1]

        buffer_start_index = self.episode_number_to_buffer_index(episode_start_index)
        return [self.step_to_s(step) for step in list(self.buffer)[buffer_start_index:buffer_index + 1]] + \
               [self.step_to_s2(self.buffer[buffer_index])]

    def step_to_s(self, step):
        return np.array(step[0])

    def step_to_a(self, step):
        return np.array(step[1])

    def step_to_r(self, step):
        return np.array(step[2])

    def step_to_t(self, step):
        return np.array(step[3])

    def step_to_s2(self, step):
        return np.array(step[4])

    # PRIVATE
    def episode_number_to_buffer_index(self, episode_number):
        return len(self.buffer) - (self.next_episode_number - episode_number) #extra (-1) cancels out in buffer and total_count

    def buffer_index_to_episode_number(self, buffer_index):
        return buffer_index -  len(self.buffer) + self.next_episode_number

    def __len__(self):
        return len(self.buffer)