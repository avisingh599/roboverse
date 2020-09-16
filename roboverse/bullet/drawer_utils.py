import pybullet as p
import roboverse.bullet.control as control
import numpy as np


def open_drawer(drawer):
    slide_drawer(drawer, -1)


def close_drawer(drawer):
    slide_drawer(drawer, 1)


def slide_drawer(drawer, direction):
    assert direction in [-1, 1]
    # -1 = open; 1 = close
    joint_names = [control.get_joint_info(drawer, j, 'jointName')
                   for j in range(p.getNumJoints(drawer))]
    drawer_frame_joint_idx = joint_names.index('base_frame_joint')

    num_ts = 15 if direction == -1 else 25

    command = np.clip(10 * direction,
                      -10 * np.abs(direction), np.abs(direction))
    # enable fast opening; slow closing

    # Wait a little before closing
    wait_ts = 0 if direction == -1 else 20
    control.step_simulation(wait_ts, t_sleep=0.01)

    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=command,
        force=10
    )

    control.step_simulation(num_ts, t_sleep=0.01)
    p.setJointMotorControl2(
        drawer,
        drawer_frame_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=10
    )
