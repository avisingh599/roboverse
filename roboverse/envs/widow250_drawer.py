from roboverse.envs.widow250 import Widow250Env
import roboverse
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
import numpy as np


class Widow250DrawerEnv(Widow250Env):

    def __init__(self,
                 **kwargs):
        self.drawer_pos = (0.5, 0.2, -.35)
        obj_pos_high = tuple(
            np.array(self.drawer_pos[:2] + (-.2,)) + np.array((0.15, 0, 0)))
        obj_pos_low = tuple(
            np.array(self.drawer_pos[:2] + (-.2,)) - np.array((-0.15, 0, 0)))
        self.drawer_opened_success_thresh = 0.8
        super(Widow250DrawerEnv, self).__init__(
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

        self.objects["drawer"] = object_utils.load_object(
            "drawer", self.drawer_pos, (0, 0, 0.707107, 0.707107), scale=0.1)
        # Open and close testing.
        self.drawer_min_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]

        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

        self.drawer_max_x_pos = object_utils.close_drawer(
            self.objects['drawer'])[0]

    def get_info(self):
        info = super(Widow250DrawerEnv, self).get_info()
        drawer_x_pos = object_utils.get_drawer_bottom_pos(
            self.objects["drawer"])[0]
        info['drawer_x_pos'] = drawer_x_pos
        info['drawer_opened_percentage'] = (
            np.abs(drawer_x_pos - self.drawer_min_x_pos) /
            (self.drawer_max_x_pos - self.drawer_min_x_pos)
        )
        return info

    def get_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(
            self.objects["drawer"])
        return handle_pos

    def is_drawer_open(self):
        info = self.get_info()
        return (info["drawer_opened_percentage"] >
                self.drawer_opened_success_thresh)

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "opening":
            return float(self.is_drawer_open())
        else:
            return super(Widow250DrawerEnv, self).get_reward(info)


if __name__ == "__main__":
    env = roboverse.make('Widow250DrawerOpen-v0',
                         gui=True, transpose_image=False)
    import time
    env.reset()
    # import IPython; IPython.embed()

    for i in range(20):
        print(i)
        obs, rew, done, info = env.step(
            np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0.]))
        print("reward", rew, "info", info)
        time.sleep(0.1)

    env.reset()
    time.sleep(1)
    for _ in range(25):
        env.step(np.asarray([0., 0., 0., 0., 0., 0., 0.6]))
        time.sleep(0.1)

    env.reset()
