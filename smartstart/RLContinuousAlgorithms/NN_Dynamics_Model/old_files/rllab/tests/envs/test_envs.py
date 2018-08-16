# import numpy as np
# from nose2 import tools
#
# from rllab.rllab.envs.box2d.car_parking_env import CarParkingEnv
# from rllab.rllab.envs.box2d.cartpole_env import CartpoleEnv
# from rllab.rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
# from rllab.rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
# from rllab.rllab.envs.box2d.mountain_car_env import MountainCarEnv
# from rllab.rllab.envs.grid_world_env import GridWorldEnv
# from rllab.rllab.envs.identification_env import IdentificationEnv
# import os
#
# MUJOCO_ENABLED = True
#
# try:
#     import mujoco_py
#     from rllab.rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
#     from rllab.rllab.envs.mujoco.hopper_env import HopperEnv
#     from rllab.rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
#     from rllab.rllab.envs.mujoco.point_env import PointEnv
#     from rllab.rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
#     from rllab.rllab.envs.mujoco.swimmer_env import SwimmerEnv
#     from rllab.rllab.envs.mujoco.walker2d_env import Walker2DEnv
#     from rllab.rllab.envs.mujoco.gather.point_gather_env import PointGatherEnv
#     from rllab.rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
#     from rllab.rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
#     from rllab.rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
#     from rllab.rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
#     from rllab.rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
# except OSError:
#     print("Warning: Mujoco not installed. Skipping mujoco-related tests")
#     MUJOCO_ENABLED = False
#
# from rllab.rllab.envs.noisy_env import NoisyObservationEnv, DelayedActionEnv
# from rllab.rllab.envs.normalized_env import NormalizedEnv
# from rllab.rllab.envs.proxy_env import ProxyEnv
# from rllab.rllab.envs.gym_env import GymEnv
#
# simple_env_classes = [
#     GridWorldEnv,
#     CartpoleEnv,
#     CarParkingEnv,
#     CartpoleSwingupEnv,
#     DoublePendulumEnv,
#     MountainCarEnv,
# ]
#
# if MUJOCO_ENABLED:
#     simple_env_classes.extend([
#         PointEnv,
#         Walker2DEnv,
#         SwimmerEnv,
#         SimpleHumanoidEnv,
#         InvertedDoublePendulumEnv,
#         HopperEnv,
#         HalfCheetahEnv,
#         PointGatherEnv,
#         SwimmerGatherEnv,
#         AntGatherEnv,
#         PointMazeEnv,
#         SwimmerMazeEnv,
#         AntMazeEnv,
#     ])
#
# envs = [cls() for cls in simple_env_classes]
# envs.append(
#     ProxyEnv(envs[0])
# )
# envs.append(
#     IdentificationEnv(CartpoleEnv, {})
# )
# envs.append(
#     NoisyObservationEnv(CartpoleEnv())
# )
# envs.append(
#     DelayedActionEnv(CartpoleEnv())
# )
# envs.append(
#     NormalizedEnv(CartpoleEnv())
# )
# envs.append(
#     GymEnv('CartPole-v0')
# )
#
#
# @tools.params(*envs)
# def test_env(env):
#     print("Testing", env.__class__)
#     ob_space = env.observation_space
#     act_space = env.action_space
#     ob = env.reset()
#     assert ob_space.contains(ob)
#     a = act_space.sample()
#     assert act_space.contains(a)
#     res = env.step(a)
#     assert ob_space.contains(res.observation)
#     assert np.isscalar(res.reward)
#     if 'CIRCLECI' in os.environ:
#         print("Skipping rendering test")
#     else:
#         env.render()
#     env.terminate()
