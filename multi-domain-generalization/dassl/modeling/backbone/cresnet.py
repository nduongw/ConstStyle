import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from dassl.modeling.ops import ConstStyle
from .build import BACKBONE_REGISTRY
from .backbone import Backbone

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CResNet(Backbone):

    def __init__(
        self, block, layers, cfg, **kwargs
    ):
        self.inplanes = 64
        super().__init__()
        # backbone network
        self.num_conststyle = cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE
        self.conststyle = [ConstStyle(i, cfg) for i in range(self.num_conststyle)]
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.cfg = cfg
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def stylemaps(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conststyle[0](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        return x
    
    def featuremaps(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE >= 1:
            x = self.conststyle[0](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer1(x)
        if self.cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE >= 2:
            x = self.conststyle[1](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer2(x)
        if self.cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE >= 3:
            x = self.conststyle[2](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer3(x)
        if self.cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE >= 4:
            x = self.conststyle[3](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer4(x)
        if self.cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE >= 5:
            x = self.conststyle[4](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        return x
        
    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        f = self.featuremaps(x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

@BACKBONE_REGISTRY.register()
def cresnet18(pretrained=True, cfg=None, **kwargs):
    model = CResNet(block=BasicBlock, layers=[2, 2, 2, 2], cfg=cfg)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

@BACKBONE_REGISTRY.register()
def cresnet50(pretrained=True, cfg=None, **kwargs):
    model = CResNet(block=Bottleneck, layers=[3, 4, 6, 3], cfg=cfg)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
