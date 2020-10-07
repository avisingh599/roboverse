from roboverse.envs.widow250 import Widow250Env
import roboverse
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
import numpy as np
import itertools


class Widow250DoubleDrawerEnv(Widow250Env):

    def __init__(self,
                 drawer_pos=(0.47, 0.2, -.35),
                 drawer_quat=(0, 0, 0.707107, 0.707107),
                 left_opening=True,  # False is not supported
                 start_opened=False,
                 **kwargs):
        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.left_opening = left_opening
        self.start_opened = start_opened
        self.drawer_opened_success_thresh = 0.95
        obj_pos_high, obj_pos_low = self.get_obj_pos_high_low()
        super(Widow250DoubleDrawerEnv, self).__init__(
            # object_names=object_names,
            # object_scales=object_scales,
            # object_orientations=object_orientations,
            object_position_high=obj_pos_high,
            object_position_low=obj_pos_low,
            **kwargs
        )

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low, self.object_position_high,
            self.num_objects,
        )
        self.original_object_positions = object_positions

        self.objects["drawer_bottom"] = object_utils.load_object(
            "drawer", self.drawer_pos, self.drawer_quat, scale=0.1)
        self.objects["drawer_top"] = object_utils.load_object(
            "drawer_no_handle", self.drawer_pos + np.array([0, 0, 0.07]), self.drawer_quat, scale=0.1)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer_bottom'])[0]

        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

        opened_drawer_x_pos = object_utils.close_drawer(
            self.objects['drawer_bottom'])[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos

        if self.start_opened:
           object_utils.open_drawer(self.objects['drawer_bottom'])

    def get_obj_pos_high_low(self):
        obj_pos_high = tuple(
            np.array(self.drawer_pos[:2] + (-.2,)) +
            (1 - 2 * (not self.left_opening)) * np.array((0.12, 0, 0)))
        obj_pos_low = tuple(
            np.array(self.drawer_pos[:2] + (-.2,)) -
            (1 - 2 * (not self.left_opening)) * np.array((-0.12, 0, 0)))
        return obj_pos_high, obj_pos_low

    def get_info(self):
        info = super(Widow250DoubleDrawerEnv, self).get_info()
        drawer_x_pos = object_utils.get_drawer_bottom_pos(
            self.objects["drawer_bottom"])[0]
        info['drawer_x_pos'] = drawer_x_pos
        info['drawer_opened_percentage'] = \
            self.get_drawer_opened_percentage()
        info['drawer_opened_success'] = info["drawer_opened_percentage"] > \
            self.drawer_opened_success_thresh
        return info

    def get_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(
            self.objects["drawer_bottom"])
        return handle_pos

    def is_drawer_open(self):
        info = self.get_info()
        return info['drawer_opened_success']

    def get_drawer_opened_percentage(self):
        drawer_x_pos = object_utils.get_drawer_bottom_pos(
            self.objects["drawer_bottom"])[0]
        return object_utils.get_drawer_opened_percentage(
            self.left_opening, self.drawer_min_x_pos,
            self.drawer_max_x_pos, drawer_x_pos)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "opening":
            return float(self.is_drawer_open())
        else:
            return super(Widow250DoubleDrawerEnv, self).get_reward(info)
