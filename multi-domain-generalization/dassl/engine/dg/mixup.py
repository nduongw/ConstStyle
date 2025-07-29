from torch.nn import functional as F
import numpy as np
import torch
import copy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import mixup

def onehot(label, n_classes, device):
    return torch.zeros(label.size(0), n_classes).to(device).scatter_(1, label.view(-1, 1), 1)

@TRAINER_REGISTRY.register()
class Mixup(TrainerX):
    """Mixup baseline."""

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        label_copy = copy.deepcopy(label)
        indices = torch.randperm(input.size(0))
        input2 = input[indices]
        label2 = label[indices]
        label = onehot(label, self.num_classes, device=self.device)
        label2 = onehot(label2, self.num_classes, device=self.device)
        
        input_mix, label_mix = mixup(input, input2, label, label2, beta=1)
        output = self.model(input_mix)
        loss = F.cross_entropy(output, label_mix)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'accuracy': compute_accuracy(output, label_copy)[0].item()
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
