import gym
from roboverse.assets.shapenet_object_lists \
    import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, PICK_PLACE_TRAIN_OBJECTS, PICK_PLACE_TEST_OBJECTS

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
        'id': 'Widow250MultiObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.20),
                   'object_position_low': (.53, .15, -.20),
                   'xyz_action_scale': 0.2,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250MultiObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.20),
                   'object_position_low': (.5, .15, -.20),
                   'xyz_action_scale': 0.2,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),

                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                       (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.30),
                   'object_position_low': (.5, .15, -.30),
                   'xyz_action_scale': 0.2,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),

                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                        (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
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
    # Pick and place environments
    {
        'id': 'Widow250PickPlace-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.59, .27, -.25),

                   'container_name': 'bowl_small',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_position_z': -0.35,

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250SinglePutInBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.59, .27, -.25),

                   'container_name': 'bowl_small',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,
                   'container_position_z': -0.35,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250SinglePutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.25),
                   'object_position_high': (.7, .27, -.25),

                   'container_name': 'bowl_small',
                   'container_position_low': (.5, 0.26, -.25),
                   'container_position_high': (.7, 0.26, -.25),
                   'container_position_z': -0.35,

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.25),
                   'object_position_high': (.7, .27, -.25),

                   'container_name': 'bowl_small',
                   'container_position_low': (.5, 0.26, -.25),
                   'container_position_high': (.7, 0.26, -.25),
                   'container_position_z': -0.35,

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTrain-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',
                   'container_position_low': (.5, 0.26, -.25),
                   'container_position_high': (.7, 0.26, -.25),
                   'container_position_z': -0.35,

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TEST_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',
                   'container_position_low': (.5, 0.26, -.25),
                   'container_position_high': (.7, 0.26, -.25),
                   'container_position_z': -0.35,

                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInTray-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.59, .27, -.25),

                   'container_name': 'tray',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.18,
                   'container_position_z': -0.37,
                   'place_success_radius_threshold': 0.04,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInTrayRandomTrayPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'tray',
                   'container_position_low': (.5, 0.25, -.25),
                   'container_position_high': (.7, 0.25, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.18,
                   'container_position_z': -0.37,
                   'place_success_radius_threshold': 0.04,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),

                   }
    },
    {
        'id': 'Widow250PutInBox-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.59, .27, -.25),

                   'container_name': 'open_box',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.1,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.1,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInBoxRandomBoxPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'open_box',
                   'container_position_low': (.5, 0.23, -.25),
                   'container_position_high': (.7, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.1,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.1,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PlaceOnCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.59, .27, -.25),

                   'container_name': 'cube',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.05,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.03,
                   'place_success_height_threshold': -0.23,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PlaceOnCubeRandomCubePosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'cube',
                   'container_position_low': (.5, 0.22, -.25),
                   'container_position_high': (.7, 0.24, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.05,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.03,
                   'place_success_height_threshold': -0.23,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInPanTefal-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'pan_tefal',
                   'container_position_low': (.70, 0.23, -.25),
                   'container_position_high': (.70, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.4,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInPanTefalRandomPanTefalPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'pan_tefal',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.24, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.4,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInTableTop-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'table_top',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.05,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutInTableTopRandomTableTopPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'table_top',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.05,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnTorus-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'torus',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (1, 1, 1, 1),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnTorusRandomTorusPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'torus',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (1, 1, 1, 1),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.05,
                   'min_distance_from_object': 0.09,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnCubeConcave-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'cube_concave',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.06,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnCubeConcaveRandomCubeConcavePosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'cube_concave',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.06,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnPlate-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'plate',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.46,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnPlateRandomPlatePosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'plate',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.46,
                   'container_position_z': -0.35,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnHusky-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'husky',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnHuskyRandomHuskyPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'husky',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.15,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnMarbleCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'marble_cube',
                   'container_position_low': (.72, 0.23, -.25),
                   'container_position_high': (.72, 0.23, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeRandomMarbleCubePosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.25),
                   'object_position_high': (.69, .27, -.25),

                   'container_name': 'marble_cube',
                   'container_position_low': (.50, 0.22, -.25),
                   'container_position_high': (.70, 0.26, -.25),
                   'container_orientation': (0, 0, 0.707107, 0.707107),
                   'container_scale': 0.07,
                   'container_position_z': -0.35,
                   'place_success_height_threshold': -0.23,
                   'place_success_radius_threshold': 0.04,
                   'min_distance_from_object': 0.10,

                   'camera_distance': 0.29,
                   'camera_target_pos': (0.6, 0.2, -0.28),
                   }
    },
    # Drawer environments
    {
        'id': 'Widow250DrawerOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerRandomizedOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerRandomizedEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    }
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
