import pybullet_data
import pybullet as p
import os
import roboverse.bullet as bullet
import importlib.util
import numpy as np

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')


def table():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.75, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id


def duck(base_position=(.65, 0.2, -.4)):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    duck_id = p.loadURDF('duck_vhacd.urdf',
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=0.8)
    return duck_id


def widow250():
    widow250_path = os.path.join(ASSET_PATH, 'interbotix_descriptions/urdf/wx250s.urdf')
    widow250_id = p.loadURDF(widow250_path,
                             basePosition=[0.6, 0, -0.4],
                             baseOrientation=bullet.deg_to_quat([180., 180., 180])
                             )
    return widow250_id

# Shapenet Utils

def import_metadata(asset_path):
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", os.path.join(asset_path, "metadata.py"))
    metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(metadata)
    return metadata.obj_path_map, metadata.path_scaling_map


def import_shapenet_metadata():
    return import_metadata(SHAPENET_ASSET_PATH)


def load_obj(filepathcollision, filepathvisual, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None):
    collisionid= p.createCollisionShape(p.GEOM_MESH, fileName=filepathcollision, meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepathvisual, meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, pos, quat)
    return body


def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5, quat=[1, -1, 0, 0]):
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(
            dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name),
        object_position,
        quat, # this rotates objects 90 degrees. Originally: [0, 0, 1, 0]
        scale=scale_local * scaling)
    return obj
