import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os
from scipy.linalg import sqrtm

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

class ConstStyle(nn.Module):
    def __init__(self, idx, cfg, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.cfg = cfg
        self.eps = eps
        self.alpha_test = cfg.TRAINER.CONSTSTYLE.ALPHA_TEST
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
    
    def store_style(self, x, domain):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain.extend(domain.detach().squeeze().cpu().numpy())
    
    def cal_mean_std(self, idx, epoch):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        if self.cfg.CLUSTER == 'domain_label':
            print('Clustering using domain label')
            unique_labels = np.unique(self.domain)
            domain_label = torch.tensor(self.domain)
            unique_domain_label = torch.unique(domain_label)
            
            means, covs = [], []
            for val in unique_domain_label:
                domain_ele = reshaped_data[domain_label == val]
                domain_bayes_cluster = BayesianGaussianMixture(n_components=1, covariance_type='full', init_params='k-means++', max_iter=200)
                domain_bayes_cluster.fit(domain_ele)
                means.append(domain_bayes_cluster.means_[0])
                covs.append(domain_bayes_cluster.covariances_[0])
            
            cluster_mean = np.mean(means, axis=0)
            cluster_cov = np.mean(covs, axis=0)
            
        else:
            print('Clustering using GMM')
            print(f'Number of cluster: {self.cfg.NUM_CLUSTERS}')
            self.bayes_cluster = BayesianGaussianMixture(n_components=self.cfg.NUM_CLUSTERS, covariance_type='full', init_params='k-means++', max_iter=200)
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
        tsne = TSNE(n_components=2, random_state=self.cfg.SEED)
        plot_data = tsne.fit_transform(reshaped_data)
        
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'{self.cfg.OUTPUT_DIR}', f'testing-features_after{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()
    
    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        if store_feature:
            self.store_style(x, domain)
        
        if (not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB) or not apply_conststyle:
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))
            
            if self.alpha_test:
                beta = (1 - self.alpha_test) * const_std + self.alpha_test * sig
                gamma = (1 - self.alpha_test) * const_mean + self.alpha_test * mu
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
