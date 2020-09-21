from roboverse.envs.widow250 import Widow250Env
import roboverse
import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.envs import objects
import numpy as np

class Widow250ButtonEnv(Widow250Env):

    def __init__(self,
                 button_pos=(0.5, 0.2, -.3),
                 button_pos_low=None,
                 button_pos_high=None,
                 button_quat=(0, 0, 0.707107, 0.707107),
                 object_names=(None,),
                 object_scales=(None,),
                 object_orientations=(None,),
                 object_position_high=(None,),
                 object_position_low=(None,),
                 target_object=None,
                 reward_type="button_pressing",
                 **kwargs):
        self.button_pos = button_pos
        self.button_pos_low = button_pos_low
        self.button_pos_high = button_pos_high
        self.button_quat = button_quat
        self.button_pressed_success_thresh = 0.8
        self.objects_on_scene = None not in object_names
        super(Widow250ButtonEnv, self).__init__(
            object_names=object_names,
            object_scales=object_scales,
            object_orientations=object_orientations,
            object_position_high=object_position_high,
            object_position_low=object_position_low,
            target_object=target_object,
            reward_type=reward_type,
            **kwargs)

    def set_button_pos(self):
        if (self.button_pos_low is not None and
                self.button_pos_high is not None):
            rand_button_pos = object_utils.generate_object_positions(
                self.button_pos_low, self.button_pos_high, 1)[0]
            return rand_button_pos
        elif self.button_pos is not None:
            return self.button_pos

    def _load_meshes(self):
        if self.objects_on_scene:
            super(Widow250ButtonEnv, self)._load_meshes()
        else:
            self.objects = {}

            self.table_id = objects.table()
            self.robot_id = objects.widow250()

        self.button_pos = self.set_button_pos()

        self.objects['button'] = object_utils.load_object(
            "button", self.button_pos, self.button_quat, scale=0.25)

        self.button_min_z_pos = object_utils.push_down_button(
            self.objects['button'])[2]
        self.button_max_z_pos = object_utils.pop_up_button(
            self.objects['button'])[2]

    def get_info(self):
        info = {}
        if self.objects_on_scene:
            info = super(Widow250ButtonEnv, self).get_info()
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
        if self.reward_type == "button_pressing":
            return float(self.is_button_pressed())
        elif self.objects_on_scene:
            return super(Widow250ButtonEnv).get_reward(info)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    env = roboverse.make('Widow250ButtonPress-v0',
                         gui=True, transpose_image=False,
                         object_names=("shed", "sack_vase"),
                         object_scales=(0.6,0.6),
                         object_orientations=((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                         object_position_high=(.7, .25, -.30),
                         object_position_low=(.6, .15, -.30),
                         target_object="shed",)
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
