from torch.nn import functional as F
import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX, SimpleNet
from dassl.metrics import compute_accuracy
from dassl.data import DataManager
from dassl.data.transforms import build_augmented_transform, MultiCounterfactualAugmentIncausal, FactualAugmentIncausal
from dassl.data import AugmentedDatasetWrapper
from torch.cuda.amp import autocast,GradScaler
from dassl.modeling.backbone import adaptor
from dassl.utils import (
    count_num_param, load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler

@TRAINER_REGISTRY.register()
class MetaCausal(TrainerX):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.check_cfg(cfg)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.adapt_criterion = nn.MSELoss()
        self.factor_num = self.cfg.TRAINER.META_CAUSAL.FACTOR_NUM
    
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        self.parameter_list = []
        cfg = self.cfg
        print('Building model')
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        # print(self.model)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        
        self.E_to_W = adaptor.effect_to_weight(7,70,1).to(self.device)
        self.parameter_list.append({'params':self.E_to_W.parameters(),'lr':cfg.OPTIM.LR})
        self.AdaptNet = []
        for i in range(cfg.TRAINER.META_CAUSAL.FACTOR_NUM):
            mapping = adaptor.mapping(self.model.backbone.out_features,1024,self.model.backbone.out_features,4).to(self.device)
            self.AdaptNet.append(mapping)
            self.parameter_list.append({'params':mapping.parameters(),'lr':cfg.OPTIM.LR})
        self.parameter_list.append({'params':self.model.parameters(),'lr':cfg.OPTIM.LR})
        
        # print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM, predefined_params=self.parameter_list)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def build_data_loader(self):
        self.dm = DataManager(self.cfg, custom_tfm_train=build_augmented_transform, dataset_wrapper=AugmentedDatasetWrapper)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
    
    def forward_backward(self, batch):
        input, input_ra, input_ca, input_fa, label = self.parse_batch_train(batch)
        var_num = len(list(range(0, 31, self.cfg.TRAINER.META_CAUSAL.STRIDE)))
        b, c, h, w = input.shape
        input_fa = input_fa.reshape(b * self.factor_num, c, h , w)
        input_ca = input_ca.reshape(b * self.factor_num * var_num, c, h, w)
        b_sample_num = label.size(0)
        label_repeat = label.unsqueeze(0).reshape(b_sample_num,1).repeat((1,self.factor_num)).reshape(1,b_sample_num*self.factor_num).squeeze()
    
        with autocast():
            features = self.model.backbone(input)
            output = self.model(input)
            features_fa = self.model.backbone(input_fa)
            features_ra = self.model.backbone(input_ra)
            output_ra = self.model(input_ra)
            output_ca = self.model(input_ca)
            
            features_repeat = features.repeat((1,self.factor_num)).reshape(features_fa.shape)
            features_adapt = torch.zeros(features_fa.shape).to(self.device)
            for b in range(b_sample_num):
                for j in range(self.factor_num):
                    features_adapt[b*self.factor_num+j] = self.AdaptNet[j](features_fa[b*self.factor_num+j])

            output_adapt = self.model.classifier(features_adapt)
            
            #learning causality
            output_ra_repeat = output_ra.repeat((1,self.factor_num*var_num)).reshape(output_ca.shape)
            effect_context = output_ra_repeat - output_ca
            effect_context = effect_context.reshape(b_sample_num,self.factor_num,var_num,-1)
            effect_context = effect_context.mean(axis=2).reshape(b_sample_num*self.factor_num,-1)
            
            weight = self.E_to_W(effect_context)
            weight = weight.reshape(b_sample_num,self.factor_num)
            alphas = F.softmax(weight,dim=1)
            
            features_ra_adapt = torch.zeros(features_ra.shape).cuda()
            for b in range(b_sample_num):
                for j in range(self.factor_num):
                    features_ra_adapt[b] = features_ra_adapt[b]+ alphas[b,j]*self.AdaptNet[j](features_ra[b])     
            output_adapt_ra = self.model.classifier(features_ra_adapt)
            
            cls_loss = self.cls_criterion(output, label)
            re_mapping = self.adapt_criterion(features_adapt, features_repeat) 
            re_causal = self.adapt_criterion(features_ra_adapt, features)                
            cls_loss_mapping = self.cls_criterion(output_adapt, label_repeat)
            cls_loss_causal = self.cls_criterion(output_adapt_ra, label)
            
            loss = cls_loss + cls_loss_mapping + self.cfg.TRAINER.META_CAUSAL.LAMBDA_RE*re_mapping + self.cfg.TRAINER.META_CAUSAL.LAMBDA_CAUSAL*(self.cfg.TRAINER.META_CAUSAL.LAMBDA_RE*re_causal + cls_loss_causal)
            
            self.model_backward_and_update(loss)

            loss_summary = {
                'loss': loss.item(),
                'accuracy': compute_accuracy(output, label)[0].item()
            }

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            return loss_summary
    
    @torch.no_grad()
    def test(self):
        self.set_model_mode('eval')
        self.evaluator.reset()
        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None
        var_num = len(list(range(0, 31, self.cfg.TRAINER.META_CAUSAL.STRIDE)))
        CA = MultiCounterfactualAugmentIncausal(self.factor_num,self.cfg.TRAINER.META_CAUSAL.STRIDE, is_test=True) 
        
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            b_sample_num = input.size(0)
            with torch.no_grad():
                features = self.model.backbone(input)
                output = self.model(input)
                input_ca = CA(input)
                input_ca = torch.from_numpy(input_ca).to(self.device, dtype=torch.float64)
                output_ca = self.model(input_ca.float())
                output_repeat = output.repeat((1,CA.factor_num*var_num)).reshape(output_ca.shape)
                effect_context = output_repeat - output_ca
                effect_context = effect_context.reshape(b_sample_num,CA.factor_num,var_num,-1)
                effect_context = effect_context.mean(axis=2).reshape(b_sample_num*CA.factor_num,-1)
                weight = self.E_to_W(effect_context)
                weight = weight.reshape(b_sample_num,CA.factor_num)
                alphas = F.softmax(weight,dim=1)
                f_adapt = torch.zeros(features.shape).to(self.device)
                for b in range(b_sample_num):
                    for j in range(CA.factor_num):
                        f_adapt[b] = f_adapt[b]+ alphas[b,j]*self.AdaptNet[j](features[b])
                p_adapt = self.model.classifier(f_adapt)
                self.evaluator.process(p_adapt, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if self.args.wandb:
                self.args.tracker.log({
                    f'test {k}': v 
                }, step=self.epoch+1)
            self.write_scalar(tag, v, self.epoch)
            

    def parse_batch_train(self, batch):
        input = batch['img']
        input_ra = batch['img_ra']
        input_ca = batch['img_ca']
        input_fa = batch['img_fa']
        label = batch['label']
        
        input = input.to(self.device)
        input_ra = input_ra.to(self.device)
        input_ca = input_ca.to(self.device)
        input_fa = input_fa.to(self.device)
        label = label.to(self.device)
        return input, input_ra, input_ca, input_fa, label