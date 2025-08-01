from .build import build_backbone, BACKBONE_REGISTRY # isort:skip
from .backbone import Backbone # isort:skip

from .vgg import vgg16
from .resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152, resnet18_ms_l1,
    resnet50_ms_l1, resnet18_ms_l12, resnet50_ms_l12, resnet101_ms_l1,
    resnet18_ms_l123, resnet50_ms_l123, resnet101_ms_l12, resnet101_ms_l123,
    resnet18_efdmix_l1, resnet50_efdmix_l1, resnet18_efdmix_l12,
    resnet50_efdmix_l12, resnet101_efdmix_l1, resnet18_efdmix_l123,
    resnet50_efdmix_l123, resnet101_efdmix_l12, resnet101_efdmix_l123
)

from .alexnet import alexnet
from .mobilenetv2 import mobilenetv2
from .wide_resnet import wide_resnet_16_4, wide_resnet_28_2, wide_resnet_16_4_mixstyle, wide_resnet_16_4_uncertainty, wide_resnet_16_4_conststyle, wide_resnet_16_4_correlated_uncertainty
from .uresnet import uresnet18, curesnet18, uresnet50, curesnet50, usresnet18, usresnet50, cusresnet18, cusresnet50
from .cnn_digitsdg import cnn_digitsdg, cnn_digitsdg_mixstyle, cnn_digitsdg_uncertainty, cnn_digitsdg_correlated_uncertainty, cnn_digitsdg_conststyle
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from .shufflenetv2 import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,
    shufflenet_v2_x2_0
)
from .cnn_digitsingle import cnn_digitsingle
from .cnn_digit5_m3sda import cnn_digit5_m3sda
from .cresnet import cresnet18, cresnet50
