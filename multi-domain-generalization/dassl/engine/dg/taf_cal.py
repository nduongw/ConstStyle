from torch.nn import functional as F
import numpy as np
import torch
import time
import datetime
import random
import math

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import (
    MetricMeter, AverageMeter
)
from dassl.metrics import compute_accuracy

def get_ampl_features(active_layers, model, dset, device):
    amp_features = {
        "amp_layer0_all": None,
        "amp_layer1_all": None,
        "amp_layer2_all": None,
        "amp_layer3_all": None
    }
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(dset):
            x = batch['img'].to(device)
            early_features = model.backbone.early_layer(x)
            layer1_features = model.backbone.layer1(early_features)
            layer2_features = model.backbone.layer2(layer1_features)
            layer3_features = model.backbone.layer3(layer2_features)
            layer4_features = model.backbone.layer4(layer3_features)
            yhat = model.backbone.classifier_layer(layer4_features)
            if active_layers[0]:
                _, layer0_amp = fft(early_features.cpu())
            if active_layers[1]:
                _, layer1_amp = fft(layer1_features.cpu())
            if active_layers[2]:
                _, layer2_amp = fft(layer2_features.cpu())
            if active_layers[3]:
                _, layer3_amp = fft(layer3_features.cpu())
            if i == 0:
                if active_layers[0]:
                    amp_features['amp_layer0_all'] = layer0_amp
                if active_layers[1]:
                    amp_features['amp_layer1_all'] = layer1_amp
                if active_layers[2]:
                    amp_features['amp_layer2_all'] = layer2_amp
                if active_layers[3]:
                    amp_features['amp_layer3_all'] = layer3_amp
            else:
                if active_layers[0]:
                    amp_features['amp_layer0_all'] = torch.cat(
                        [amp_features['amp_layer0_all'], layer0_amp])
                if active_layers[1]:
                    amp_features['amp_layer1_all'] = torch.cat(
                        [amp_features['amp_layer1_all'], layer1_amp])
                if active_layers[2]:
                    amp_features['amp_layer2_all'] = torch.cat(
                        [amp_features['amp_layer2_all'], layer2_amp])
                if active_layers[3]:
                    amp_features['amp_layer3_all'] = torch.cat(
                        [amp_features['amp_layer3_all'], layer3_amp])
    if active_layers[0]:
        amp_features['amp_layer0_mean'] = torch.mean(
            amp_features['amp_layer0_all'], dim=0).unsqueeze(0)
    if active_layers[1]:
        amp_features['amp_layer1_mean'] = torch.mean(
            amp_features['amp_layer1_all'], dim=0).unsqueeze(0)
    if active_layers[2]:
        amp_features['amp_layer2_mean'] = torch.mean(
            amp_features['amp_layer2_all'], dim=0).unsqueeze(0)
    if active_layers[3]:
        amp_features['amp_layer3_mean'] = torch.mean(
            amp_features['amp_layer3_all'], dim=0).unsqueeze(0)

    return amp_features

def mixup_func(x, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(torch.get_device(x))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def swap_amp(features, amp_features, layer, orig_amp_factor, device):
    phase, amp = fft(features, device=device)
    amp_mean = amp_features[f'amp_layer{layer}_mean'].to(device).repeat(
        features.shape[0], 1, 1, 1)

    amp = (orig_amp_factor * amp) + (1 - orig_amp_factor) * amp_mean
    features = ifft(phase, amp, device)
    return features

def fft(img, device='cpu', eps=True, eps_val=1e-7):
    # get fft of the image
    fft_img = torch.view_as_real(torch.fft.fft2(img, norm='backward'))
    # extract phase and amplitude from the fft
    phase, ampl = extract_phase_amlp(fft_img.clone(), eps=eps, eps_val=eps_val)

    return phase, ampl


def extract_phase_amlp(fft_img, eps=True, eps_val=1e-7):
    # fft_img: size should be batch_size * 3 * h * w * 2
    ampl = fft_img[:, :, :, :, 0]**2 + fft_img[:, :, :, :, 1]**2
    if eps:
        ampl = torch.sqrt(ampl + eps)
        phase = torch.atan2(fft_img[:, :, :, :, 1] + eps_val,
                            fft_img[:, :, :, :, 0] + eps_val)
    else:
        ampl = torch.sqrt(ampl)
        phase = torch.atan2(fft_img[:, :, :, :, 1], fft_img[:, :, :, :, 0])
    return phase, ampl


def ifft(phase, ampl, device='cpu'):
    # recompse fft of image
    fft_img = torch.zeros(
        (phase.shape[0], phase.shape[1], phase.shape[2], phase.shape[3], 2),
        dtype=torch.float).to(device)
    fft_img[:, :, :, :, 0] = torch.cos(phase) * ampl
    fft_img[:, :, :, :, 1] = torch.sin(phase) * ampl

    # get the recomposed image
    _, _, imgH, imgW = phase.size()
    # image = torch.irfft(fft_img,
    #                     signal_ndim=2,
    #                     onesided=False,
    #                     signal_sizes=[imgH, imgW])
    complex_tensor = torch.complex(fft_img[..., 0], fft_img[..., 1])
    image = torch.fft.ifft2(complex_tensor, s=[imgH, imgW], norm='backward')
    return image.real

def mixup_amp(args, features, device):
    phase, mixed_amp = fft(features, device)
    if random.random() > args.TRAINER.TAFCAL.MIX_FUNC_RATIO:
        mixed_amp = mixup_func(mixed_amp, args.TRAINER.TAFCAL.MIXUP_ALPHA)

    amp_idx = torch.randperm(mixed_amp.shape[0])
    mixed_amp = mixed_amp[amp_idx].view(mixed_amp.size())

    features = ifft(phase, mixed_amp, device)
    return features

@TRAINER_REGISTRY.register()
class TAFCAL(TrainerX):
    """TAF Cal"""
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.check_cfg(cfg)
        self.mix_layers = [0, 0, 1, 0]
        self.active_layers = [0, 0, 1, 0]
    
    def train(self):
        """Generic training loops."""
        self.first_round_max_epoch = math.ceil(self.max_epoch * 0.7)

        self.before_train()
        for self.epoch in range(self.start_epoch, self.first_round_max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        
        print('Starting further training')
        self.amp_features = get_ampl_features(self.active_layers, self.model, self.train_loader_x, self.device)
        
        for self.epoch in range(self.first_round_max_epoch, self.max_epoch):
            self.before_epoch()
            self.further_train()
            self.further_after_epoch()
        
        self.after_train()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        if random.random() > (1 - self.cfg.TRAINER.TAFCAL.MIX_AMP_RATIO) and (self.epoch >= self.cfg.TRAINER.TAFCAL.WARM_UP_EPOCH):
            do_swap_amp = True
        else:
            do_swap_amp = False
        early_features = self.model.backbone.early_layer(input)
        if do_swap_amp and self.mix_layers[0]:
            early_features = mixup_amp(self.cfg, early_features, self.device)
        layer1_features = self.model.backbone.layer1(early_features)
        
        if do_swap_amp and self.mix_layers[1]:
            layer1_features = mixup_amp(self.cfg, layer1_features, self.device)
        layer2_features = self.model.backbone.layer2(layer1_features)
        if do_swap_amp and self.mix_layers[2]:
            layer2_features = mixup_amp(self.cfg, layer2_features, self.device)
        layer3_features = self.model.backbone.layer3(layer2_features)
        if do_swap_amp and self.mix_layers[3]:
            layer3_features = mixup_amp(self.cfg, layer3_features, self.device)
        layer4_features = self.model.backbone.layer4(layer3_features)
        output_feats = self.model.backbone.classifier_layer(layer4_features)
        output = self.model.classifier(output_feats)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss, do_swap_amp)

        loss_summary = {
            'loss': loss.item(),
            'accuracy': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_backward_and_update(self, loss, do_swap_amp, names=None):
        self.model_zero_grad(names)
        # if do_swap_amp and (self.epoch >= self.cfg.TRAINER.TAFCAL.WARM_UP_EPOCH):
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.model_backward(loss)
        self.model_update(names)
            
    def further_train(self):
        if self.epoch > 0 and self.cfg.TRAINER.TAFCAL.KEEP_UPDATE_AMP_MEAN:
            del self.amp_features
            self.amp_features = get_ampl_features(self.mix_layers, self.model, self.train_loader_x, self.device)
        
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.further_forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
                if self.args.wandb:
                    self.args.tracker.log({
                        f'training {name}': meter.avg 
                    }, step=self.epoch+1)
                    
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
    
    def further_forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        ft_amp_factor = 0.5
        if random.random() > (1 - self.cfg.TRAINER.TAFCAL.SWAP_AMP_RATIO):
            do_swap_amp = True
            if self.cfg.TRAINER.TAFCAL.CALIBRATE_ALPHA != -1:
                ft_amp_factor = np.random.beta(self.cfg.TRAINER.TAFCAL.CALIBRATE_ALPHA, self.cfg.TRAINER.TAFCAL.CALIBRATE_ALPHA)
        else:
            do_swap_amp = False
        if (random.random() > (1 - self.cfg.TRAINER.TAFCAL.MIX_AMP_RATIO)) and self.cfg.TRAINER.TAFCAL.MIXUP_AMP_FEATURES:
            do_mix_amp = True
        else:
            do_mix_amp = False
        
        layer0_features = self.model.backbone.early_layer(input)
        if do_swap_amp and self.active_layers[0]:
            layer0_features = swap_amp(layer0_features, self.amp_features, 0, ft_amp_factor, self.device)
        if do_mix_amp and self.mix_layers[0]:
            layer0_features = mixup_amp(self.cfg, layer0_features, self.device)
        layer1_features = self.model.backbone.layer1(layer0_features)

        if do_swap_amp and self.active_layers[1]:
            layer1_features = swap_amp(layer1_features, self.amp_features, 1, ft_amp_factor, self.device)
        if do_mix_amp and self.mix_layers[1]:
            layer1_features = mixup_amp(self.cfg, layer1_features, self.device)
        layer2_features = self.model.backbone.layer2(layer1_features)

        if do_swap_amp and self.active_layers[2]:
            layer2_features = swap_amp(layer2_features, self.amp_features, 2, ft_amp_factor, self.device)
        if do_mix_amp and self.mix_layers[2]:
            layer2_features = mixup_amp(self.cfg, layer2_features, self.device)
        layer3_features = self.model.backbone.layer3(layer2_features)
        
        if do_swap_amp and self.active_layers[3]:
            layer3_features = swap_amp(layer3_features, self.amp_features, 3, ft_amp_factor, self.device)
        if do_mix_amp and self.mix_layers[3]:
            layer3_features = mixup_amp(self.cfg, layer3_features, self.device)
        layer4_features = self.model.backbone.layer4(layer3_features)
        
        output_feats = self.model.backbone.classifier_layer(layer4_features)
        output = self.model.classifier(output_feats)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss, do_swap_amp)

        loss_summary = {
            'loss': loss.item(),
            'accuracy': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def further_after_epoch(self):
        not_last_epoch = (self.epoch + 1) != self.max_epoch
        do_test = self.cfg.TEST.EVAL_FREQ > 0 and not self.cfg.TEST.NO_TEST
        meet_test_freq = (
            self.epoch + 1
        ) % self.cfg.TEST.EVAL_FREQ == 0 if do_test else False
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if not_last_epoch and do_test and meet_test_freq:
            results = self.test_swap_amp()
            if results['accuracy'] > self._best_acc:
                print(f'Test accuracy increases from {round(self._best_acc, 2)} to {round(results["accuracy"], 2)} --> save model')
                self.save_model(self.epoch, self.output_dir)
                self._best_acc = results['accuracy']

        if not_last_epoch and meet_checkpoint_freq:
            self.save_model(self.epoch, self.output_dir)
    
    def test_swap_amp(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        test_amp_factor = self.cfg.TRAINER.TAFCAL.TEST_AMP_FACTOR

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            layer0_features = self.model.backbone.early_layer(input)
            if self.active_layers[0]:
                layer0_features = swap_amp(layer0_features, self.amp_features, 0,
                                        test_amp_factor, self.device)
            layer1_features = self.model.backbone.layer1(layer0_features)
            if self.active_layers[1]:
                layer1_features = swap_amp(layer1_features, self.amp_features, 1,
                                        test_amp_factor, self.device)
            layer2_features = self.model.backbone.layer2(layer1_features)
            if self.active_layers[2]:
                layer2_features = swap_amp(layer2_features, self.amp_features, 2,
                                        test_amp_factor, self.device)
            layer3_features = self.model.backbone.layer3(layer2_features)
            if self.active_layers[3]:
                layer3_features = swap_amp(layer3_features, self.amp_features, 3,
                                        test_amp_factor, self.device)
            layer4_features = self.model.backbone.layer4(layer3_features)
            output_feats = self.model.backbone.classifier_layer(layer4_features)
            output = self.model.classifier(output_feats)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if self.args.wandb:
                self.args.tracker.log({
                    f'test {k}': v 
                }, step=self.epoch+1)
            self.write_scalar(tag, v, self.epoch)

        return results
        
    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
