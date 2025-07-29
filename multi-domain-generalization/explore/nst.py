import net
from pathlib import Path
import torch.nn as nn
import argparse
import torch
import wandb
from tqdm import tqdm
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from explore.util import *
    
args = get_args()
cfg = setup_cfg(args)

if args.wandb:
    if 'u' in cfg.MODEL.BACKBONE.NAME:
        job_type = 'DSU'
    elif 'c' in cfg.MODEL.BACKBONE.NAME:
        job_type = 'ConstStyle'
    else:
        job_type = 'Baseline'
    if cfg.MODEL.BACKBONE.PRETRAINED:
        job_type += '-pretrained'
        
    tracker = wandb.init(
        project = 'StyleDG',
        entity = 'aiotlab',
        config = args,
        group = f'{cfg.DATASET.NAME}',
        name = f'train={cfg.DATASET.SOURCE_DOMAINS}_test={cfg.DATASET.TARGET_DOMAINS}_type={args.option}',
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

trainer = build_trainer(cfg, args)

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

network = net.Net(vgg, decoder)
network.train()
network.to(device)

train_loader_x = trainer.train_loader_x
test_loader = trainer.test_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_iter = iter(cycle(train_loader_x))
test_iter = iter(cycle(test_loader))

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(100000)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(train_iter)
    style_images = next(test_iter)
    content_images, content_label = trainer.parse_batch_test(content_images)
    style_images, style_label = trainer.parse_batch_test(style_images)
    
    loss_c, loss_s = network(content_images, style_images)
    loss_c = 1.0 * loss_c
    loss_s = 10.0 * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'\nContent loss: {loss_c.item()} | Style loss: {loss_s.item()}')

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.save_model_interval:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'{args.save_dir}/decoder_iter_{i+1}.pth.tar')