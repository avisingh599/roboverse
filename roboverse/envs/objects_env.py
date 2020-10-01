import gym
import numpy as np
import argparse
import os

from roboverse.bullet.serializable import Serializable
import roboverse.bullet as bullet
from roboverse.envs import objects
from roboverse.bullet import object_utils
# from .multi_object import MultiObjectEnv
import roboverse
from PIL import Image

from roboverse.assets.shapenet_object_lists import (
    PICK_PLACE_TRAIN_OBJECTS,
    TRAIN_CONTAINERS,
    TEST_CONTAINERS,
    OBJECT_SCALINGS,
    CONTAINER_CONFIGS,
)

objects_to_visualize = (PICK_PLACE_TRAIN_OBJECTS +
                        TRAIN_CONTAINERS + TEST_CONTAINERS)


def get_scaling(object_name):
    obj_scale = OBJECT_SCALINGS.get(object_name, None)
    container_info = CONTAINER_CONFIGS.get(object_name, None)
    if obj_scale is not None:
        return obj_scale
    elif container_info is not None:
        return 0.4 * container_info['container_scale']
    else:
        return 0.75


class ObjectsEnv(gym.Env, Serializable):

    def __init__(self,
                 control_mode='continuous',
                 observation_mode='pixels',
                 observation_img_h=128 * (5 + 2),
                 observation_img_w=128 * (9 + 2),
                 transpose_image=True,

                 object_names=tuple(objects_to_visualize),
                 object_scales=tuple([
                    get_scaling(object_name) for object_name
                    in objects_to_visualize]),
                 object_orientations=tuple(
                    [(0, 0, 1, 0)] * len(objects_to_visualize)),
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 target_object=objects_to_visualize[0],
                 load_tray=True,

                 num_sim_steps=10,
                 num_sim_steps_reset=10,
                 num_sim_steps_discrete_action=75,

                 reward_type='grasping',
                 grasp_success_height_threshold=-0.25,
                 grasp_success_object_gripper_threshold=0.1,

                 xyz_action_scale=0.2,
                 abc_action_scale=20.0,
                 gripper_action_scale=20.0,

                 ee_pos_high=(0.8, .4, -0.1),
                 ee_pos_low=(.4, -.2, -.34),
                 camera_target_pos=(0.75, -0.1, -0.28),
                 camera_distance=0.425,
                 camera_roll=0.0,
                 camera_pitch=-90,
                 camera_yaw=180,

                 gui=False,
                 in_vr_replay=False,
                 ):

        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_h = observation_img_h
        self.observation_img_w = observation_img_w
        self.transpose_image = transpose_image

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = \
            grasp_success_object_gripper_threshold

        self.gui = gui

        # TODO(avi): Add limits to ee orientation as well
        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)

        # object stuff
        assert target_object in object_names
        assert len(object_names) == len(object_scales)
        self.load_tray = load_tray
        self.num_objects = len(object_names)
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.object_names = object_names
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_name in \
                zip(object_orientations, object_scales, self.object_names):
            self.object_orientations[object_name] = orientation
            self.object_scales[object_name] = object_scale

        self.in_vr_replay = in_vr_replay
        # self._load_meshes()

        # self.movable_joints = bullet.get_movable_joints(self.robot_id)
        # self.end_effector_index = END_EFFECTOR_INDEX
        # self.reset_joint_values = RESET_JOINT_VALUES
        # self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale
        self.gripper_action_scale = gripper_action_scale

        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_roll = camera_roll
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance,
                                yaw=self.camera_yaw,
                                pitch=self.camera_pitch,
                                roll=self.camera_roll,
                                up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_h, self.observation_img_w)

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True  # TODO(avi): Clean this up

        self.reset()
        # self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
        #     self.robot_id, self.end_effector_index)

    def _load_meshes(self):
        self.table_id = objects.table()
        # self.robot_id = objects.widow250()

        self.objects = {}
        dist = 0.1
        x_low, y_low = 0.35, -.3
        num_obj_per_row = 9
        for i, object_name in enumerate(self.object_names):
            obj_pos_x = x_low + (dist * (i % num_obj_per_row))
            obj_pos_y = y_low + (dist * (i // num_obj_per_row))
            obj_pos_z = -0.36
            object_position = (obj_pos_x, obj_pos_y, obj_pos_z)
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
        bullet.step_simulation(self.num_sim_steps_reset)

    def _set_action_space(self):
        self.action_dim = 0
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (
                self.observation_img_h * self.observation_img_w) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def reset(self):
        self._load_meshes()

    def step(self, action):
        return self.render_obs(), None, None, None

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.observation_img_h, self.observation_img_w,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=0)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img


if __name__ == "__main__":
    env = roboverse.make('PickPlaceTrainObject-v0',
                         gui=True, transpose_image=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    for i in range(10):
        img, _, _, _ = env.step(None)
        im = Image.fromarray(img)
        im.save(os.path.join(args.save_path, '{}.png'.format(i)))
