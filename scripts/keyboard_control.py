import numpy as np
import time

import roboverse
from roboverse.envs.widow250 import Widow250Env
import roboverse.bullet as bullet

env = roboverse.make('Widow250MultiTaskGrasp-v0', gui=True)

keys_pressed = {bullet.p.B3G_LEFT_ARROW: np.array([0.1, 0, 0, 0, 0, 0, 0]),
bullet.p.B3G_RIGHT_ARROW: np.array([-0.1, 0, 0, 0, 0, 0, 0]),
bullet.p.B3G_UP_ARROW: np.array([0, -0.1, 0, 0, 0, 0, 0]),
bullet.p.B3G_DOWN_ARROW: np.array([0, 0.1, 0, 0, 0, 0, 0]),
ord('j'): np.array([0, 0, 0.2, 0, 0, 0, 0]),
ord('k'): np.array([0, 0, -0.2, 0, 0, 0, 0]),
ord('h'): np.array([0, 0, 0, 0, 0, 0, -0.7]),
ord('l'): np.array([0, 0, 0, 0, 0, 0, 0.7])}

while True:
    pressed = False
    action = np.array([0, 0, 0, 0, 0, 0, 0], dtype='float32')
    keys = bullet.p.getKeyboardEvents()
    for qKey in keys_pressed.keys():
        if qKey in keys:
            action += keys_pressed[qKey]
            pressed = True
    if pressed:
        env.step(action)
    time.sleep(0.1)
