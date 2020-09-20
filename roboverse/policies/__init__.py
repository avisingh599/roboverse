from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
)
