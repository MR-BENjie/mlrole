REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .qrelation_controller import QrelationMAC
from .group_controller import  GroupMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY["qrealtion_mac"] = QrelationMAC
REGISTRY["group_mac"] = GroupMAC