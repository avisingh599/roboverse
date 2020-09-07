import pybullet_data
import pybullet as p
import os
import importlib.util
import numpy as np
from .control import get_object_position, get_link_state

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')

MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS = 100


def check_grasp(object_name,
                object_id_map,
                robot_id,
                end_effector_index,
                grasp_success_height_threshold,
                grasp_success_object_gripper_threshold,
                ):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    success = False
    if object_height > grasp_success_height_threshold:
        ee_pos, _ = get_link_state(
            robot_id, end_effector_index)
        object_gripper_distance = np.linalg.norm(
            object_pos - ee_pos)
        if object_gripper_distance < \
                grasp_success_object_gripper_threshold:
            success = True

    return success


def generate_object_positions(object_position_low, object_position_high,
                              num_objects, min_distance=0.07):
    object_positions = np.random.uniform(
        low=object_position_low, high=object_position_high)
    object_positions = np.reshape(object_positions, (1, 3))
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while object_positions.shape[0] < num_objects:
        i += 1
        object_position_candidate = np.random.uniform(
            low=object_position_low, high=object_position_high)
        object_position_candidate = np.reshape(
            object_position_candidate, (1, 3))
        min_distance_so_far = []
        for o in object_positions:
            dist = np.linalg.norm(o - object_position_candidate)
            min_distance_so_far.append(dist)
        min_distance_so_far = np.array(min_distance_so_far)
        if (min_distance_so_far > min_distance).any():
            object_positions = np.concatenate(
                (object_positions, object_position_candidate), axis=0)

        if i > max_attempts:
            ValueError('Min distance could not be assured')

    return object_positions


def import_metadata(asset_path):
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", os.path.join(asset_path, "metadata.py"))
    metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(metadata)
    return metadata.obj_path_map, metadata.path_scaling_map


def import_shapenet_metadata():
    return import_metadata(SHAPENET_ASSET_PATH)


# TODO(avi, albert) This should be cleaned up
shapenet_obj_path_map, shapenet_path_scaling_map = import_shapenet_metadata()


def load_obj(filepath_collision, filepath_visual, pos=[0, 0, 0],
             quat=[0, 0, 0, 1], scale=1):
    collisionid = p.createCollisionShape(p.GEOM_MESH,
                                         fileName=filepath_collision,
                                         meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepath_visual,
                                   meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, pos, quat)
    return body


def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5,
                         quat=[1, -1, 0, 0]):
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(
            dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/'
                              'models/model_normalized.obj'.format(
            dir_name, object_name),
        object_position,
        quat,  # this rotates objects 90 degrees. Originally: [0, 0, 1, 0]
        scale=scale_local * scaling)
    return obj


def set_obj_scalings(object_names, scalings):
    obj_path_map, path_scaling_map = \
        dict(shapenet_obj_path_map), dict(shapenet_path_scaling_map)

    object_path_dict = dict(
        [(obj, path) for obj, path in obj_path_map.items() if
         obj in object_names])
    scaling_map = dict(
        [(name, scaling * path_scaling_map[
            '{}/{}'.format(obj_path_map[name].split("/")[-2],
                           obj_path_map[name].split("/")[-1])])
         for name, scaling in zip(object_names, scalings)])

    return object_path_dict, scaling_map


def set_pos_high_low_maps(object_names, pos_high_list, pos_low_list):
    pos_high_map = {}
    pos_low_map = {}
    for object_name, pos_high, pos_low in \
            zip(object_names, pos_high_list, pos_low_list):
        assert np.all(np.array(pos_low) <= np.array(pos_high))
        pos_high_map[object_name] = pos_high
        pos_low_map[object_name] = pos_low

    return pos_high_map, pos_low_map
