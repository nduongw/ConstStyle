from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import torch.utils.model_zoo as model_zoo

__all__ = ['uresnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean) * self.factor
        sqrtvar_std = self.sqrtvar(std) * self.factor

        beta = self.reparameterize(mean, sqrtvar_mu)
        gamma = self.reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class CorrelatedDistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                _, mu_eng_vector = torch.linalg.eigh(C*correlation_mu+self.eps*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)

            try:
                _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)

            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)

        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig

        return x_normed * sig_mix + mu_mix


class BasicBlock(nn.Module):  # for 18 and 34
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

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
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class UResNet(nn.Module):
    """Residual network.

    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, block, layers, num_classes=0, last_stride=1,
                 fc_dims=0, pooling=None, dropout_p=None, pertubration=None,
                 **kwargs):
        self.inplanes = 64
        super(UResNet, self).__init__()
        self.feature_dim = 512 * block.expansion
        self.fc_dims = fc_dims
        if fc_dims > 0:
            self.feature_dim = fc_dims

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print(pertubration())
        self.pertubration0 = pertubration()
        self.pertubration1 = pertubration()
        self.pertubration2 = pertubration()
        self.pertubration3 = pertubration()
        self.pertubration4 = pertubration()
        self.pertubration5 = pertubration()

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], IN=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, IN=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, IN=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, IN=False)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(2048, self.feature_dim)
        self.BN = nn.BatchNorm1d(self.feature_dim)
        self.BN.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1, IN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, IN=IN))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.pertubration1(x)
        x = self.layer1(x)
        x = self.pertubration2(x)
        x = self.layer2(x)
        x = self.pertubration3(x)
        x = self.layer3(x)
        x = self.pertubration4(x)
        x = self.layer4(x)
        x = self.pertubration5(x)
        return x

    def forward(self, x, feature_withbn=False):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc_dims > 0:
            v = self.fc(v)

        bn_v = self.BN(v)

        if not self.training:
            return F.normalize(bn_v)

        y = self.classifier(bn_v)

        if feature_withbn:
            return bn_v, y

        return v, y


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def uresnet50(num_classes, num_features=0, pretrained=True, **kwargs):
    model = UResNet(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=num_features,
        dropout_p=None,
        pertubration=DistributionUncertainty,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def curesnet50(num_classes, num_features=0, pretrained=True, **kwargs):
    model = UResNet(
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=num_features,
        dropout_p=None,
        pertubration=CorrelatedDistributionUncertainty,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
