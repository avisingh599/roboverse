import numpy as np
import pybullet as p

GRAVITY = -10


def connect_headless(gui=False):
    if gui:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=0.8,
                                 cameraYaw=180,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0.6, 0, -0.4])
    p.setGravity(0, 0, GRAVITY)

