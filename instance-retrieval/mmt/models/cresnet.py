from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torch
import torch.utils.model_zoo as model_zoo

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import copy
from sklearn.manifold import TSNE
import os
from scipy.linalg import sqrtm

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

def wasserstein_distance_multivariate(mean1, cov1, mean2, cov2):
    mean_diff = mean1 - mean2
    mean_distance = np.dot(mean_diff, mean_diff)
    sqrt_cov1 = sqrtm(cov1)
    if np.iscomplexobj(sqrt_cov1):
        sqrt_cov1 = sqrt_cov1.real
    # Compute the term involving the covariance matrices
    cov_sqrt_product = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    if np.iscomplexobj(cov_sqrt_product):
        cov_sqrt_product = cov_sqrt_product.real

    cov_term = np.trace(cov1 + cov2 - 2 * cov_sqrt_product)
    wasserstein_distance = np.sqrt(mean_distance + cov_term)
    return wasserstein_distance

def mahalanobis_distance(point, mean, cov):
    inv_cov_matrix = torch.linalg.inv(cov)
    diff = point - mean
    distance = torch.sqrt(torch.matmul(torch.matmul(diff, inv_cov_matrix), torch.t(diff)))
    distance = torch.diagonal(distance)
    return distance

def euclid_distance(point, mean):
    diff = point - mean
    square_diff = diff ** 2
    sum_square_diff = torch.sum(square_diff, dim=1)
    distance = torch.sqrt(sum_square_diff)
    return distance

def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    mu2, sigma2 = torch.squeeze(mu2), torch.squeeze(sigma2)
    mean_diff_squared = (mu1 - mu2) ** 2
    std_diff_squared = (sigma1 - sigma2) ** 2
    wasserstein_distance = torch.sqrt(torch.sum(mean_diff_squared, dim=1) + torch.sum(std_diff_squared, dim=1))
    return wasserstein_distance

def sigmoid_increase(x, k=6.0):
    return 1 / (1 + torch.exp(-k * (x - 1)))

class ConstStyle5(nn.Module):
    def __init__(self, idx, args, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.args = args
        self.eps = eps
        self.alpha = args.alpha
        self.mean = []
        self.std = []
        self.domain = []
        self.const_mean = None
        self.const_cov = None
        self.bayes_cluster = None
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = (var + self.eps).sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        return mu, var
    
    def store_style(self, x):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
    
    def cal_mean_std(self, idx, epoch):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        print(f'Number of cluster: {self.args.num_cluster}')
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.args.num_cluster, covariance_type='full', init_params='k-means++', max_iter=200)
        self.bayes_cluster.fit(reshaped_data)
        
        labels = self.bayes_cluster.predict(reshaped_data)
        unique_labels, _ = np.unique(labels, return_counts=True)
        
        cluster_mean = np.mean([self.bayes_cluster.means_[i] for i in range(len(unique_labels))], axis=0)
        cluster_cov = np.mean([self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))], axis=0)
        
        self.const_mean = torch.from_numpy(cluster_mean)
        self.const_cov = torch.from_numpy(cluster_cov)
        self.generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix=self.const_cov)

    def plot_style_statistics(self, idx, epoch):
        domain_list = np.array(self.domain_list)
        #clustering
        mean_list = copy.copy(self.mean_after)
        std_list = copy.copy(self.std_after)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        classes = ['in domain 1', 'in domain 2', 'in domain 3', 'out domain']
        tsne = TSNE(n_components=2, random_state=self.args.seed)
        plot_data = tsne.fit_transform(reshaped_data)
        
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'{self.args.output_dir}', f'testing-features_after{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
    
    def calculate_domain_distance(self, idx):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        domain_list = np.array(self.domain)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        unique_domain = np.unique(domain_list)
        print(f'Unique domain: {unique_domain}')
        seen_domain_mean, seen_domain_cov = [], []
        unseen_domain_mean, unseen_domain_cov = [], []
        for val in unique_domain:
            domain_feats = reshaped_data[domain_list == val]
            domain_distribution = BayesianGaussianMixture(n_components=1, covariance_type='full', init_params='k-means++', max_iter=200)
            domain_distribution.fit(domain_feats)
            if val < 10:
                seen_domain_mean.append(domain_distribution.means_[0])
                seen_domain_cov.append(domain_distribution.covariances_[0])
            else:
                unseen_domain_mean.append(domain_distribution.means_[0])
                unseen_domain_cov.append(domain_distribution.covariances_[0])
        
        total_distance = 0.0
        for i in range(len(seen_domain_mean)):
            for j in range(len(unseen_domain_mean)):
                distance = wasserstein_distance_multivariate(unseen_domain_mean[j], unseen_domain_cov[j], seen_domain_mean[i], seen_domain_cov[i])
                total_distance += distance
        
        print(f'Total distance from seen to unseen domain of layer {idx}: {total_distance}')
        # center_to_unseen_dist = wasserstein_distance_multivariate(unseen_domain_mean[0], unseen_domain_cov[0], self.const_mean.numpy(), self.const_cov.numpy())
        # print(f'Distance from center to unseen domain of layer {idx}: {center_to_unseen_dist}\n')

    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False, apply_rate=0.0):
        if store_feature:
            self.store_style(x)
        
        if (not is_test and np.random.random() > self.args.prob) or not apply_conststyle:
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        # style_feats = torch.hstack((mu, sig)).squeeze().detach().cpu()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
            # wass_dist = wasserstein_distance(const_mean, const_std, mu, sig)
            # if self.idx == 0:
            #     print(f'Wass distance max value of conststyle layer {self.idx}: {torch.max(wass_dist)} | min value: {torch.min(wass_dist)}')

            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))
            if not apply_rate:
                beta = const_std * self.alpha + (1 - self.alpha) * sig
                gamma = const_mean * self.alpha + (1 - self.alpha) * mu
            else:
                beta = const_std
                gamma = const_mean
            
        else:
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = self.generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])
            
            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()
            
            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

            beta = const_std
            gamma = const_mean
        out = x_normed * beta + gamma
            
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

class CResNet(nn.Module):
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

    def __init__(self, block, layers, args=None, num_classes=0, last_stride=1,
                 fc_dims=0, pooling=None, dropout_p=None, pertubration=None,
                 **kwargs):
        self.inplanes = 64
        super(CResNet, self).__init__()
        self.feature_dim = 512 * block.expansion
        self.fc_dims = fc_dims
        if fc_dims > 0:
            self.feature_dim = fc_dims
        self.args = args
        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], IN=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, IN=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, IN=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, IN=False)

        self.num_conststyle = args.num_conststyle
        print('Aapplying conststyle ver5...')
        self.conststyle = [ConstStyle5(i, args) for i in range(self.num_conststyle)]
        
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

    def featuremaps(self, x, store_feature=False, apply_conststyle=False, is_test=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.args.num_conststyle >= 1:
            x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer1(x)
        if self.args.num_conststyle >= 2:
            x = self.conststyle[1](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer2(x)
        if self.args.num_conststyle >= 3:
            x = self.conststyle[2](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer3(x)
        if self.args.num_conststyle >= 4:
            x = self.conststyle[3](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer4(x)
        if self.args.num_conststyle >= 5:
            x = self.conststyle[4](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        return x

    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False, feature_withbn=False):
        f = self.featuremaps(x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        v = self.global_avgpool(f)
        v.view(v.size(0), -1)

        if self.fc_dims > 0:
            v = self.fc(v)
        
        v = torch.squeeze(v)
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

def cresnet50(num_classes, args, num_features=0, pretrained=True, **kwargs):
    model = CResNet(
        args=args,
        num_classes=num_classes,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=num_features,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
