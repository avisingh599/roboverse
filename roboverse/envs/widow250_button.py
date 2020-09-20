from roboverse.envs.widow250 import Widow250Env
import roboverse
from roboverse.bullet import object_utils
from roboverse.envs import objects
import numpy as np


class Widow250ButtonEnv(Widow250Env):

    def __init__(self,
                 button_pos=(0.5, 0.2, -.3),
                 button_quat=(0, 0, 0.707107, 0.707107),
                 **kwargs):
        self.button_pos = button_pos
        self.button_quat = button_quat
        self.button_pressed_success_thresh = 0.8
        super(Widow250ButtonEnv, self).__init__(
            object_names=(None,),
            object_scales=(None,),
            object_orientations=(None,),
            object_position_high=(None,),
            object_position_low=(None,),
            target_object=None,
            **kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        self.objects['button'] = object_utils.load_object(
            "button", self.button_pos, self.button_quat, scale=0.25)

        self.button_min_z_pos = object_utils.push_down_button(
            self.objects['button'])[2]
        self.button_max_z_pos = object_utils.pop_up_button(
            self.objects['button'])[2]

    def get_info(self):
        info = {}
        info['button_z_pos'] = self.get_button_pos()[2]
        info['button_pressed_percentage'] = (
            (self.button_max_z_pos - info['button_z_pos']) /
            (self.button_max_z_pos - self.button_min_z_pos))
        info['button_pressed_success'] = float(
            info['button_pressed_percentage'] >
            self.button_pressed_success_thresh)
        return info

    def get_button_pos(self):
        return object_utils.get_button_cylinder_pos(
            self.objects['button'])

    def is_button_pressed(self):
        info = self.get_info()
        return bool(info['button_pressed_percentage'] >
                    self.button_pressed_success_thresh)

    def get_reward(self, info):
        return float(self.is_button_pressed())


if __name__ == "__main__":
    env = roboverse.make('Widow250ButtonPress-v0',
                         gui=True, transpose_image=False)
    import time
    env.reset()
    # import IPython; IPython.embed()

    for j in range(5):
        object_utils.pop_up_button(env.objects['button'])
        time.sleep(1)
        object_utils.push_down_button(env.objects['button'])
        time.sleep(1)
        object_utils.pop_up_button(env.objects['button'])
        for i in range(20):
            obs, rew, done, info = env.step(
                np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0.]))
            print("reward", rew, "info", info)
            time.sleep(0.1)
        env.reset()
