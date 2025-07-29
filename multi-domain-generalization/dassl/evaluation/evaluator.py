import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.best_acc = 0
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # print(f'total: {self._total}')
        # softmax = nn.Softmax(dim=1)
        # score = softmax(mo)
        pred = mo.max(1)[1]
        # for i in range(len(score)):
        #     if pred[i] != gt[i]:
        #         print(f'Score :{score[i].max().item()} - Pred: {pred[i].item()} | {gt[i].item()}')
        matched = pred.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)
    
    def process2(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        matched = mo.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(mo.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)
    
    def process_multiple(self, mo, gt):
        mean_pred = torch.zeros_like(mo[0])
        mean_pred = torch.zeros_like(mo[0])
        for val in mo:
            pred = self.softmax(val)
            max_idx, max_val = pred.max(1)[1], pred.max(1)[0]
            print(f'Indices coresspond with highest value: {max_idx}, {max_val}')
            print(f'Prediction of output: {val}\n')
            mean_pred += pred
        
        final_pred = mean_pred.max(1)[1]
        print(f'Final prediction: {final_pred}\n')
        print(f'Groundtruth: {gt}\n')
        matched = final_pred.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(final_pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        self.best_acc = max(self.best_acc, acc)
        err = 100. - acc
        results['accuracy'] = acc
        results['error_rate'] = err
        results['best_accuracy'] = self.best_acc

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* current accuracy: {:.2f}%\n'
            '* best accuracy: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, self.best_acc, err)
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print('=> per-class result')
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100. * correct / total
                accs.append(acc)
                print(
                    '* class: {} ({})\t'
                    'total: {:,}\t'
                    'correct: {:,}\t'
                    'acc: {:.2f}%'.format(
                        label, classname, total, correct, acc
                    )
                )
            print('* average: {:.2f}%'.format(np.mean(accs)))

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize='true'
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, 'cmat.pt')
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
