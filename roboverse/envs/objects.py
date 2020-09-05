import pybullet_data
import pybullet as p
import os
import roboverse.core.bullet as bullet

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')


def table():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.75, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id


def duck():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    duck_id = p.loadURDF('duck_vhacd.urdf',
                         basePosition=[.75, .0, -.3],
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
