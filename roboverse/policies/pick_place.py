import numpy as np
import roboverse.bullet as bullet


class PickPlace:

    def __init__(self, env, pick_height_thresh=-0.31):
        self.env = env
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            + np.random.normal(scale=0.01)
        self.xyz_action_scale = 7.0
        self.reset()

    def reset(self):
        self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.place_attempted = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.env.target_object])
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

