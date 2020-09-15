import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'Widow250Grasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'beer_bottle',
                   'load_tray': True,
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250MultiTaskGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.20),
                   'object_position_low': (.53, .15, -.20),
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250SingleObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.20),
                   'object_position_low': (.53, .15, -.20),
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250PickPlace-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.20),
                   'object_position_high': (.59, .27, -.20),

                   'container_name': 'bowl_small',
                   'container_position': (.72, 0.23, -.35),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in ENVIRONMENT_SPECS)

    return gym_ids


def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env