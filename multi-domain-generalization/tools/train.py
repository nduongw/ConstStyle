import argparse
import torch
import wandb

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    
    if args.batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
        cfg.DATALOADER.TRAIN_U.BATCH_SIZE = args.batch_size

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.cluster:
        cfg.CLUSTER = args.cluster
    
    if args.pretrained_style:
        cfg.PRETRAINED_STYLE = args.pretrained_style    

    if args.num_clusters:
        cfg.NUM_CLUSTERS = args.num_clusters
    
    if args.distance:
        cfg.DISTANCE = args.distance
    
    if args.prob:
        cfg.TRAINER.CONSTSTYLE.PROB = args.prob

    if args.reduce:
        cfg.REDUCE = args.reduce
    
    cfg.TRAINER.CONSTSTYLE.ALPHA_TEST = args.alpha_test
    
    if args.conststyle_type:
        cfg.TRAINER.CONSTSTYLE.TYPE = args.conststyle_type
    
    if args.num_conststyle:
        cfg.TRAINER.CONSTSTYLE.NUM_CONSTSTYLE = args.num_conststyle

    cfg.MODEL.UNCERTAINTY = args.uncertainty

def setup_cfg(args):
    cfg = get_cfg_default()
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    reset_cfg(cfg, args)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if args.wandb:
        if cfg.DATASET.NAME == 'DigitsDG' or cfg.DATASET.NAME == 'CIFAR10C' or cfg.DATASET.NAME == 'DigitSingle':
            if 'mixstyle' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'MixStyle'
            elif 'correlated' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'CSU'
            elif 'uncertainty' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'DSU'
            elif 'conststyle' in cfg.MODEL.BACKBONE.NAME:
                job_type = f'ConstStyle'
            elif 'efd' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'EFDMix'
            elif cfg.TRAINER.NAME == 'RIDG':
                job_type = 'RIDG'
            elif cfg.TRAINER.NAME == 'DDAIG':
                job_type = 'DDAIG'
            elif cfg.TRAINER.NAME == 'Mixup':
                job_type = 'Mixup'
            elif cfg.TRAINER.NAME == 'Cutmix':
                job_type = 'Cutmix'
            elif cfg.TRAINER.NAME == 'CrossGrad':
                job_type = 'CrossGrad'
            else:
                job_type = 'ERM'
            if cfg.TRAINER.NAME == 'ConstStyleTrainerPretrained':
                job_type += '_pretrained_style'
        else:
            if 'uresnet' in cfg.MODEL.BACKBONE.NAME and len(cfg.MODEL.BACKBONE.NAME) == 9:
                job_type = 'DSU'
            elif 'cresnet' in cfg.MODEL.BACKBONE.NAME:
                job_type = f'ConstStyle'
            elif 'curesnet' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'CSU'
            elif 'ms_l12' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'MixStyle'
            elif 'efdmix' in cfg.MODEL.BACKBONE.NAME:
                job_type = 'EFDMix'
            elif cfg.TRAINER.NAME == 'RIDG':
                job_type = 'RIDG'
            elif cfg.TRAINER.NAME == 'DDAIG':
                job_type = 'DDAIG'
            elif cfg.TRAINER.NAME == 'Mixup':
                job_type = 'Mixup'
            elif cfg.TRAINER.NAME == 'Cutmix':
                job_type = 'Cutmix'
            elif cfg.TRAINER.NAME == 'CrossGrad':
                job_type = 'CrossGrad'
            elif cfg.TRAINER.NAME == 'MetaCausal':
                job_type = 'MetaCausal'
            elif cfg.TRAINER.NAME == 'TAFCAL':
                job_type = 'TAFCAL'
            else:
                job_type = 'Baseline'
            
            if cfg.TRAINER.NAME == 'ConstStyleTrainerPretrained':
                job_type += '_pretrained_style'

        if cfg.DATASET.NAME == 'CIFAR10C':
            process_name = f'train={cfg.DATASET.SOURCE_DOMAINS}_test={cfg.DATASET.TARGET_DOMAINS}-{cfg.DATASET.CIFAR_C_LEVEL}'
        else:
            process_name = f'train={cfg.DATASET.SOURCE_DOMAINS}_test={cfg.DATASET.TARGET_DOMAINS}'
            
        tracker = wandb.init(
            project = 'StyleDG',
            config = args,
            group = f'{cfg.DATASET.NAME}',
            name = process_name,
            job_type = job_type
        )
        args.tracker = tracker

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print_args(args, cfg)

    trainer = build_trainer(cfg, args)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    parser.add_argument('--uncertainty', default=0.0, type=float)
    parser.add_argument('--pos', nargs='+', type=int, default=[],
                        help='pos for uncertainty')
    parser.add_argument('--wandb', default=1, type=int, help='visualize on Wandb')
    parser.add_argument('--pretrained_style', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--option', default='', type=str, help='additional options')
    parser.add_argument('--update_interval', default=25, type=int, help='update cluster interval')
    parser.add_argument('--cluster', default='ot', type=str, help='cluster choosing method')
    parser.add_argument('--num_clusters', default = 3, type = int, help='number of clusters')
    parser.add_argument('--distance', default = 'wass', type = str, help='distance metric')
    parser.add_argument('--dynamic_func', default = 'loga', type = str, help='type of degradation function')
    parser.add_argument('--reduce', default = 1, type = int, help = 'reduction factor of data')
    parser.add_argument('--prob', default = 0.5, type = float)
    parser.add_argument('--mixstyle_prob', default = 0.5, type = float)
    parser.add_argument('--alpha_train', default = 0.5, type = float)
    parser.add_argument('--alpha_test', default = 0.5, type = float)
    parser.add_argument('--conststyle_type', default = 'ver1', type = str, help='distance metric')
    parser.add_argument('--num_conststyle', default = 3, type = int, help='number of integrated conststyle layers')
    parser.add_argument('--cutmix_prob', default = 0.5, type = float, help='cutmix probability')
    
    args = parser.parse_args()
    main(args)
