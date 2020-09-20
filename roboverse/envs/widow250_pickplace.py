from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv


class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 container_position_low=(.72, 0.23, -.35),
                 container_position_high=(.72, 0.23, -.35),
                 container_position_z=-0.35,
                 container_orientation=(0, 0, 0.707107, 0.707107),
                 container_scale=0.07,
                 min_distance_from_object=0.1,

                 place_success_height_threshold=-0.32,
                 place_success_radius_threshold=0.03,

                 **kwargs
                 ):
        self.container_name = container_name
        self.container_position_low = container_position_low
        self.container_position_high = container_position_high
        self.container_position_z = container_position_z
        self.container_orientation = container_orientation
        self.container_scale = container_scale
        self.min_distance_from_object = min_distance_from_object

        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold

        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        """
        TODO(avi) This needs to be cleaned up, generate function should only 
                  take in (x,y) positions instead. 
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if self.num_objects == 2:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        elif self.num_objects == 1:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_single(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        else:
            raise NotImplementedError

        # TODO(avi) Need to clean up
        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def get_reward(self, info):
        reward = float(info['place_success_target'])
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info


class Widow250PickPlaceMultiObjectEnv(MultiObjectEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceMultiObjectMultiContainerEnv(
    MultiObjectMultiContainerEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":

    # Fixed container position
    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed', 'two_handled_vase'),
        object_scales=(0.7, 0.6),
        target_object='shed',
        load_tray=False,
        object_position_low=(.49, .18, -.20),
        object_position_high=(.59, .27, -.20),

        container_name='cube',
        container_position_low=(.72, 0.23, -.20),
        container_position_high=(.72, 0.23, -.20),
        container_position_z=-0.34,
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.05,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    # env = Widow250PickPlaceEnv(
    #     reward_type='pick_place',
    #     control_mode='discrete_gripper',
    #     object_names=('shed',),
    #     object_scales=(0.7,),
    #     target_object='shed',
    #     load_tray=False,
    #     object_position_low=(.5, .18, -.25),
    #     object_position_high=(.7, .27, -.25),
    #
    #     container_name='bowl_small',
    #     container_position_low=(.5, 0.26, -.25),
    #     container_position_high=(.7, 0.26, -.25),
    #     container_orientation=(0, 0, 0.707107, 0.707107),
    #     container_scale=0.07,
    #
    #     camera_distance=0.29,
    #     camera_target_pos=(0.6, 0.2, -0.28),
    #     gui=True
    # )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)
