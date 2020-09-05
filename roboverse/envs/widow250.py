import gym
import numpy as np

from roboverse.core.serializable import Serializable
import roboverse.core.bullet as bullet


END_EFFECTOR_INDEX = 8
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57]


class Widow250Env(gym.Env, Serializable):

    def __init__(self,
                 observation_mode='pixels',
                 observation_img_dim=48,
                 num_sim_steps=10,
                 transpose_image=True,
                 gui=False,
                 camera_target_pos=[0.6, 0.0, -0.4],
                 camera_distance=0.5,
                 camera_pitch=-40,
                 camera_yaw=180,
                 ):

        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim
        self.num_sim_steps = num_sim_steps
        self.transpose_image = transpose_image
        self.gui = gui

        bullet.connect_headless(self.gui)

        self._load_meshes()
        self.movable_joints = bullet.get_movable_joints(self.robot_id)
        self.end_effector_index = END_EFFECTOR_INDEX
        self.reset_joint_values = RESET_JOINT_VALUES


        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance, yaw=self.camera_yaw,
                                pitch=self.camera_pitch, roll=0, up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_dim, self.observation_img_dim)

        self.xyz_action_scale = 1.0
        self.abc_action_scale = 20.0
        self.gripper_action_scale = 20.0

        self.reset()

    def _load_meshes(self):
        from roboverse.envs import objects
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.duck_id = objects.duck()

    def reset(self):
        bullet.reset_robot(
            self.robot_id,
            list(range(len(self.reset_joint_values))),
            self.reset_joint_values)
        # TODO(avi): reset objects
        return self.get_observation()

    def step(self, action):
        action = np.clip(action, -1, +1)  # TODO maybe clean this up


        xyz_action = action[:3]  # ee position actions
        abc_action = action[3:6]  # ee orientation actions
        gripper_action = action[6]

        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])


        target_ee_pos = ee_pos + self.xyz_action_scale*xyz_action
        ee_deg = bullet.quat_to_deg(ee_quat)
        target_ee_deg = ee_deg + self.abc_action_scale*abc_action
        target_ee_quat = bullet.deg_to_quat(target_ee_deg)
        target_gripper_state = gripper_state + [-self.gripper_action_scale*gripper_action,
                                                +self.gripper_action_scale*gripper_action]
        bullet.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state, self.robot_id,
            self.end_effector_index, self.movable_joints,
            num_sim_steps=self.num_sim_steps)

        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return self.get_observation(), reward, done, info

    def get_observation(self):
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_tips_distance = [0.]   # TODO(avi) fix this
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        if self.observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            observation = {
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_tips_distance)),
                'image': image_observation
            }
        else:
            raise NotImplementedError

        return observation

    def get_reward(self, info):
        pass

    def get_info(self):
        pass

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.observation_img_dim, self.observation_img_dim,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=0)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img


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
