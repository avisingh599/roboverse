from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp
from .button_press import ButtonPress
from .drawer_open_transfer import DrawerOpenTransfer
from .drawer_open_suboptimal import DrawerOpenSuboptimal

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
    button_press=ButtonPress,
    drawer_open_transfer=DrawerOpenTransfer
)

suboptimal_polices = dict(
    drawer_open_suboptimal=DrawerOpenSuboptimal,
)

policies.update(suboptimal_polices)
