import copy
import time

# import gym envs
import gym
import numpy as np
import tensorflow as tf


def add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[0]):
        if(std_of_noise[j]>0):
            data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data

def perform_rollouts(policy, num_rollouts, steps_per_rollout, visualize_rollouts, CollectSamples, env, dt_steps,
                     dt_from_xml):
    #collect training data by performing rollouts
    print("Beginning to do ", num_rollouts, " rollouts.")
    c = CollectSamples(env, policy, visualize_rollouts, dt_steps, dt_from_xml)
    states, controls, starting_states, rewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.")
    return states, controls, starting_states, rewards_list


def create_env(which_agent):

    # setup environment
    if(which_agent==0):
        env = gym.make('InvertedPendulum-v2')
    elif(which_agent==1):
        env = gym.make('Ant-v2')
    elif(which_agent==2):
        env = gym.make('Swimmer-v2') #dt 0.001 and frameskip=150
    elif(which_agent==3):
        env = gym.make('Reacher-v2')
    elif(which_agent==4):
        env = gym.make('HalfCheetah-v2')
    elif(which_agent==5):
        env = gym.make('Humanoid-v2') #this is a personal vrep env
    elif(which_agent==6):
        env=gym.make('Hopper-v2')
    elif(which_agent==7):
        env=gym.make('Walker2d-v2')

    #get dt value from env
    if(which_agent==5):
        dt_from_xml = env.VREP_DT
    else:
        dt_from_xml = env.env.model.opt.timestep
    print("\n\n the dt is: ", dt_from_xml, "\n\n")

    #set vars
    tf.set_random_seed(2)
    # gym.logger.setLevel(gym.logging.WARNING)
    dimO = env.observation_space.shape
    dimA = env.action_space.shape
    print ('--------------------------------- \nState space dimension: ', dimO)
    print ('Action space dimension: ', dimA, "\n -----------------------------------")

    return env, dt_from_xml


def visualize_rendering(starting_state, list_of_actions, env_inp, dt_steps, dt_from_xml, which_agent):
    env=copy.deepcopy(env_inp)

    if(which_agent==5):
        env.reset()
    else:
        # env.reset(starting_state)
        env.reset()

    for action in list_of_actions:

        if(action.shape[0]==1):
            env.step(action[0], collectingInitialData=False)
        else:
            env.step(action, collectingInitialData=False)

        if(which_agent==5):
            junk=1
        else:
            env.render()
            time.sleep(dt_steps*dt_from_xml)

    print("Done rendering.")
    return