from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp
from .button_press import ButtonPress
from .drawer_open_transfer import DrawerOpenTransfer
from .drawer_open_transfer_suboptimal import DrawerOpenTransferSuboptimal
from .drawer_close_open_transfer import DrawerCloseOpenTransfer
from .drawer_close_open_transfer_suboptimal import DrawerCloseOpenTransferSuboptimal

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
    button_press=ButtonPress,
    drawer_open_transfer=DrawerOpenTransfer,
    drawer_close_open_transfer=DrawerCloseOpenTransfer,
)

suboptimal_polices = dict(
    drawer_open_transfer_suboptimal=DrawerOpenTransferSuboptimal,
    drawer_close_open_transfer_suboptimal=DrawerCloseOpenTransferSuboptimal,
)

policies.update(suboptimal_polices)
