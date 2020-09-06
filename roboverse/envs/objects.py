import pybullet_data
import pybullet as p
import os
import roboverse.bullet as bullet

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')


def table():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.75, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id


def duck(base_position=(.65, 0.2, -.3)):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    duck_id = p.loadURDF('duck_vhacd.urdf',
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=0.8)
    return duck_id


def tray(base_position=(.60, 0.2, -.37)):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    tray_id = p.loadURDF('tray/tray.urdf',
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=0.5)
    return tray_id


def widow250():
    widow250_path = os.path.join(ASSET_PATH, 'interbotix_descriptions/urdf/wx250s.urdf')
    widow250_id = p.loadURDF(widow250_path,
                             basePosition=[0.6, 0, -0.4],
                             baseOrientation=bullet.deg_to_quat([180., 180., 180])
                             )
    return widow250_id
