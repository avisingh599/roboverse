import numpy as np
import roboverse.bullet as bullet


class DrawerOpenTransfer:

    def __init__(self, env):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_pos_thresh = 0.04
        self.ending_pos = [6.00122578e-01,  1.54955496e-01, -1.73008145e-01]
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.drawer_never_opened = True
        offset_coeff = (-1) ** (1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        done = False

        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()):
            # print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.5]
        elif (gripper_handle_dist > self.gripper_dist_thresh
                and not self.env.is_drawer_open()):
            # moving down toward handle
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.5]
        elif not self.env.is_drawer_open():
            # print("opening drawer")
            x_command = (-1) ** (1 - self.env.left_opening)
            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.5]
        elif (abs(ee_pos[2] - self.ending_pos[2]) >
                self.ending_pos_thresh):
            self.drawer_never_opened = False
            action_xyz = (self.ending_pos - ee_pos)* 2.0   # force upward action
            action_xyz[2] *= 3.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.5]
        elif (np.linalg.norm(ee_pos - self.ending_pos) >
                self.ending_pos_thresh):
            print("Lift upward")
            self.drawer_never_opened = False
            action_xyz = (self.ending_pos - ee_pos) * 3.0  # force upward action
            print(np.linalg.norm(ee_pos - self.ending_pos))
            print(ee_pos - self.ending_pos)
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [-0.5]
        else:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.5]

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        if self.env.use_neutral_action:
            action = np.concatenate((action, neutral_action))
        return action, agent_info
