from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp
from .button_press import ButtonPress
from.drawer_grasp import DrawerGrasp

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
    button_press=ButtonPress,
    drawer_grasp=DrawerGrasp
)
