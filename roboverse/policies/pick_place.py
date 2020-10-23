import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS
from .drawer_open_transfer import DrawerOpenTransfer


class PickPlace:

    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=7.0,
                 pick_point_noise=0.00, drop_point_noise=0.00):
        self.env = env
        self.pick_height_thresh_noisy = (
            pick_height_thresh + np.random.normal(scale=0.01))
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point[2] = -0.32
        self.drop_point = self.env.container_position
        self.drop_point[2] = -0.2
        self.place_attempted = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point  - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_droppoint_dist > 0.02:
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        neutral_action = [0.]
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class PickPlaceOpen:

    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=7.0,
                 pick_point_z=-0.32, suboptimal=False):
        self.env = env
        self.pick_height_thresh_noisy = (
            pick_height_thresh + np.random.normal(scale=0.01))
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_z = pick_point_z
        self.suboptimal = suboptimal

        self.drawer_policy = DrawerOpenTransfer(env, suboptimal=self.suboptimal)

        self.reset()

    def reset(self):
        self.pick_point = bullet.get_object_position(self.env.blocking_object)[0]
        self.pick_point[2] = self.pick_point_z
        self.drop_point = bullet.get_object_position(self.env.tray_id)[0]
        self.drop_point[2] = -0.2

        if self.suboptimal and np.random.uniform() > 0.5:
            self.drop_point[0] += np.random.uniform(-0.2, 0.0)
            self.drop_point[1] += np.random.uniform(0.0, 0.2)

        self.place_attempted = False
        self.neutral_taken = False
        self.drawer_policy.reset()

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(self.env.blocking_object)
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False
        neutral_action = [0.]

        if self.place_attempted:
            # Return to neutral, then open the drawer.
            if self.neutral_taken:
                action, info = self.drawer_policy.get_action()
                action_xyz = action[:3]
                action_angles = action[3:6]
                action_gripper = [action[6]]
                neutral_action = [action[7]]
                done = info['done']
            else:
                action_xyz = [0., 0., 0.]
                action_angles = [0., 0., 0.]
                action_gripper = [0.0]
                neutral_action = [0.7]
                self.neutral_taken = True
        elif gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point  - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_droppoint_dist > 0.02:
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class PickPlaceOpenSuboptimal(PickPlaceOpen):
    def __init__(self, env, **kwargs):
        super(PickPlaceOpenSuboptimal, self).__init__(
            env, suboptimal=True, **kwargs,
        )


class PickPlaceOld:

    def __init__(self, env, pick_height_thresh=-0.31):
        self.env = env
        self.pick_height_thresh_noisy = (
            pick_height_thresh + np.random.normal(scale=0.01))
        self.xyz_action_scale = 7.0
        self.reset()

    def reset(self):
        self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.place_attempted = False
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)

        container_pos = self.env.container_position
        target_pos = np.append(container_pos[:2], container_pos[2] + 0.15)
        target_pos = target_pos + np.random.normal(scale=0.01)
        gripper_target_dist = np.linalg.norm(target_pos - ee_pos)
        gripper_target_threshold = 0.03

        done = False

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif object_gripper_dist > self.dist_thresh and self.env.is_gripper_open:
            # move near the object
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_target_dist > gripper_target_threshold:
            # lifted, now need to move towards the container
            action_xyz = (target_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
