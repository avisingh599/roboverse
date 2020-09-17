from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils


class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 container_position=(.72, 0.23, -.35),
                 container_orientation=(0, 0, 0.707107, 0.707107),
                 container_scale=0.07,

                 place_success_height_threshold=-0.32,
                 place_success_radius_threshold=0.03,

                 **kwargs
                 ):
        self.container_name = container_name
        self.container_position = container_position
        self.container_orientation = container_orientation
        self.container_scale = container_scale

        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold

        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        super(Widow250PickPlaceEnv, self)._load_meshes()
        object_utils.load_object(self.container_name,
                                 self.container_position,
                                 self.container_orientation,
                                 self.container_scale)

    def get_reward(self, info):
        reward = float(info['place_success_target'])
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info