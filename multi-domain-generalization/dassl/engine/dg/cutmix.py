from torch.nn import functional as F
import numpy as np
import torch

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

@TRAINER_REGISTRY.register()
class Cutmix(TrainerX):
    """Cutmix baseline."""
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.beta = 1.0
        self.cutmix_prob = args.cutmix_prob

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        
        r = np.random.rand(1)
        if self.beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = self.model(input)
            loss = F.cross_entropy(output, target_a) * lam + F.cross_entropy(output, target_b) * (1. - lam)
        else:
            output = self.model(input)
            loss = F.cross_entropy(output, label)
        
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'accuracy': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
