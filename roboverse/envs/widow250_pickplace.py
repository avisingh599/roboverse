from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet


class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 container_position_low=(.72, 0.23, -.35),
                 container_position_high=(.72, 0.23, -.35),
                 container_position_z=-0.35,
                 container_orientation=(0, 0, 0.707107, 0.707107),
                 container_scale=0.07,

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

        self.place_success_height_threshold = place_success_height_threshold
        self.place_success_radius_threshold = place_success_radius_threshold

        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        super(Widow250PickPlaceEnv, self)._load_meshes()
        """
        TODO(avi) This needs to be cleaned up, generate function should only 
                  take in (x,y) positions instead. 
        """
        assert self.container_position_low[2] == \
               self.original_object_positions[0, 2]

        self.container_position = object_utils.generate_object_positions(
            self.container_position_low,
            self.container_position_high,
            len(self.object_names) +1,  # +1 is for the container itself
            min_distance=0.1,
            current_positions=self.original_object_positions,
        )[-1, :]

        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

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


if __name__ == "__main__":

    # Fixed container position
    # env = Widow250PickPlaceEnv(
    #     reward_type='pick_place',
    #     control_mode='discrete_gripper',
    #     object_names=('shed',),
    #     object_scales=(0.7,),
    #     target_object='shed',
    #     load_tray=False,
    #     object_position_low=(.49, .18, -.20),
    #     object_position_high=(.59, .27, -.20),
    #
    #     container_name='bowl_small',
    #     container_position_low=(.72, 0.23, -.35),
    #     container_position_high=(.72, 0.23, -.35),
    #     container_orientation=(0, 0, 0.707107, 0.707107),
    #     container_scale=0.07,
    #
    #     camera_distance=0.29,
    #     camera_target_pos=(0.6, 0.2, -0.28),
    #     gui=True
    # )

    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed',),
        object_scales=(0.7,),
        target_object='shed',
        load_tray=False,
        object_position_low=(.5, .18, -.25),
        object_position_high=(.7, .27, -.25),

        container_name='bowl_small',
        container_position_low=(.5, 0.26, -.25),
        container_position_high=(.7, 0.26, -.25),
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.07,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)