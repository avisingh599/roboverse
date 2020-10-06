from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp
from .button_press import ButtonPress
from .drawer_open_transfer import DrawerOpenTransfer
from .drawer_open_suboptimal import DrawerOpenSuboptimal
from .drawer_open_transfer_suboptimal import DrawerOpenTransferSuboptimal

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
    button_press=ButtonPress,
    drawer_open_transfer=DrawerOpenTransfer
)

suboptimal_polices = dict(
    drawer_open_suboptimal=DrawerOpenSuboptimal,
    drawer_open_transfer_suboptimal=DrawerOpenTransferSuboptimal,
)

policies.update(suboptimal_polices)
