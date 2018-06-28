""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import pickle

class ReplayBuffer(object):
    """
    A buffer with a First in First Out policy,
    Stores vectors of (state, action, reward, terminal, new_state),
    Upon calling the start_new_episode method creates a marker to mark where said episode begins
    Can create sample batches (random entries from the buffer)
    NOTE: the add method only allows the self.main_agent from adding entries, this way if multiple agents try to add
            to the replay buffer, only one state entry will add (this is the case with Smart Starts)
    """

    def __init__(self, main_agent, max_buffer_size, random_seed=123):
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
        if random_seed is not None:
            random.seed(random_seed)

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

    def clear(self):
        self.buffer.clear()
        self.next_episode_number = 0

    def start_new_episode(self, state):
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

    # PRIVATE
    def episode_number_to_buffer_index(self, episode_count_index):
        return len(self.buffer) - (self.next_episode_number - episode_count_index) #extra (-1) cancels out in buffer and total_count
