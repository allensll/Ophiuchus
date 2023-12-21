from .lenet import lenet5
from .minionn import minionn
# from .resnet_sm import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from .preresnet_sm import preactresnet20, preactresnet32, preactresnet44, preactresnet56, preactresnet110, preactresnet1202
from .fixup_resnet_sm import *
from .resnet_sm import *
from .resnet import *

from .layers import PConv2d, PLinear, PBatchNorm2d