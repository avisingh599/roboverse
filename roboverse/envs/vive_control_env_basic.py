import gym
import numpy as np

from roboverse.bullet.serializable import Serializable
import roboverse.bullet as bullet
from roboverse.envs.widow250 import Widow250Env

if __name__ == "__main__":
    env = Widow250Env(gui=True)
    import time

    for i in range(25):
        print(i)
        env.step(np.asarray([0., 0., 0., 0., 0., 0., -0.5]))
        time.sleep(0.1)

    env.reset()
    for _ in range(25):
        env.step(np.asarray([0., 0., 0., 0., 0., 0., +0.5]))
        time.sleep(0.1)

    env.reset()