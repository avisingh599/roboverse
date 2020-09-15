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
SHAPENET_SCALE = 0.5


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


def load_shapenet_object(object_name, object_position,
                            object_quat=(1, -1, 0, 0),  scale=1.0):
    object_path = shapenet_obj_path_map[object_name]
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    filepath_collision = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name, object_name))
    filepath_visual = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name))
    scale = SHAPENET_SCALE * scale * shapenet_path_scaling_map[object_path]
    collisionid = p.createCollisionShape(p.GEOM_MESH,
                                         fileName=filepath_collision,
                                         meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepath_visual,
                                   meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, object_position, object_quat)
    return body
