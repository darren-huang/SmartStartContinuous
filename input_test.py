from pynput.keyboard import Key, Listener, KeyCode
from smartstart.environments.continuous_mountain_car_editted import Continuous_MountainCarEnv
from gym.wrappers.time_limit import TimeLimit
import time

def get_on_press_release():
    action = 0
    max_action = 1
    def on_press(key):
        nonlocal action
        # print(key)
        if key == KeyCode.from_char('a'):
            action = -max_action
            # print("go left")
        if key == key == KeyCode.from_char('d'):
            action = max_action
            # print('go right')

    def on_release(key):
        nonlocal action
        if key == key == KeyCode.from_char('a'):
            action = 0
            # print("stop")
        if key == key == KeyCode.from_char('d'):
            action = 0
            # print('stop')
        if key == Key.esc:
            # Stop listener
            return False

    def get_action():
        return action
    return on_press, on_release, get_action

if __name__ == "__main__":
    # Collect events until released

    import random
    import gym
    from smartstart.reinforcementLearningCore.rlTrain import rlTrain
    from smartstart.utilities.plot import plot_summary, show_plot, \
        mean_reward_episode, steps_episode

    # configuring environment
    # ENV_NAME = 'MountainCarContinuous-v0'
    # env = gym.make(ENV_NAME)
    untimed_env = Continuous_MountainCarEnv(.4)
    env = TimeLimit(untimed_env,
                    max_episode_steps=None,
                    max_episode_seconds=None)

    on_press, on_release, get_action = get_on_press_release()
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:

        step, new_observation, done = 0, None, False
        env.reset()
        while not done:
            old_observation = new_observation
            action = get_action()
            new_observation, reward, done, info = env.step([action])
            print("Step: {}, State: {}, Action: {}, New_State: {}, Reward: {}  ||  ".format(step, old_observation, action, new_observation, reward).replace("\n", ""))
            env.env.render()
            step += 1
