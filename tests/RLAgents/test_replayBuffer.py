from unittest import TestCase
from smartstart.RLAgents.replay_buffer import ReplayBuffer


class TestReplayBuffer(TestCase):
    def test_add_normal(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.add(0, 1,1,1,True,1)
        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(1)
        assert s_batch[0] == 1
        assert a_batch[0] == 1
        assert r_batch[0] == 1
        assert t_batch[0] == True
        assert s2_batch[0] == 1

    def test_size(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.add(0, 1, 1, 1, True, 1)
        assert replay_buffer.size() == 1

    def test_size_max(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        replay_buffer.add(0, 1, 1, 1, True, 1)
        assert replay_buffer.size() == 1

    def test_size_max_replace(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        replay_buffer.add(0, 1, 1, 1, True, 1)
        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(1)
        assert s_batch[0] == 1
        assert a_batch[0] == 1
        assert r_batch[0] == 1
        assert t_batch[0] == True
        assert s2_batch[0] == 1

    def test_episodes(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.start_new_episode(0)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        assert replay_buffer.next_episode_number == 1
        assert replay_buffer.episode_starting_indices[0] == 0
        assert len(replay_buffer.episode_starting_indices) == 1

    def test_episodes_multiple(self):
        for i in range(1, 21):
            replay_buffer = ReplayBuffer(0, i)
            for _ in range(i):
                replay_buffer.start_new_episode(0)
                replay_buffer.add(0, 2, 2, 2, False, 2)
            assert replay_buffer.next_episode_number == i
            for index in range(i):
                assert replay_buffer.episode_starting_indices[index] == index
            assert len(replay_buffer.episode_starting_indices) == i

    def test_episode_removal(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.start_new_episode(0)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        assert replay_buffer.next_episode_number == 1 # reindexing
        assert len(replay_buffer.episode_starting_indices) == 0

    def test_multiple_same_episodes(self):
        replay_buffer = ReplayBuffer(0, 1)
        replay_buffer.start_new_episode(0)
        replay_buffer.start_new_episode(0)
        assert replay_buffer.episode_starting_indices[0] == 0
        assert len(replay_buffer.episode_starting_indices) == 1

    def test_multiple_episodes(self):
        replay_buffer = ReplayBuffer(0, 2)
        replay_buffer.start_new_episode(0)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        replay_buffer.start_new_episode(0)
        assert replay_buffer.episode_starting_indices[0] == 0
        assert replay_buffer.episode_starting_indices[1] == 1
        assert len(replay_buffer.episode_starting_indices) == 2

    def test_episode_removal_indexing(self):
        replay_buffer = ReplayBuffer(0, 2)
        replay_buffer.start_new_episode(0)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        replay_buffer.start_new_episode(0)
        replay_buffer.add(0, 2, 2, 2, False, 2)
        # should have episodes [0,1] after adding another item, should reduce to [1] which reindexes to [0]
        replay_buffer.add(0, 2, 2, 2, False, 2)
        assert replay_buffer.next_episode_number == 2  # reindexing
        assert len(replay_buffer.episode_starting_indices) == 1
        assert replay_buffer.episode_starting_indices[0] == 0

    def test_episode_number_to_buffer_index(self):
        replay_buffer = ReplayBuffer(0, 10)
        for i in range(20):
            replay_buffer.start_new_episode(0)
            replay_buffer.add(0, i, i, i, False, i)
            for j in range(len(replay_buffer.episode_starting_indices)):
                episode_index = replay_buffer.episode_starting_indices[-(j+1)] # going backwards
                buffer_index = replay_buffer.episode_number_to_buffer_index(episode_index)
                assert (i-j, i-j, i-j, False, i-j) == replay_buffer.buffer[buffer_index]

