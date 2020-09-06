import pybullet_data
import pybullet as p
import os
import importlib.util
import numpy as np

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')


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
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
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
