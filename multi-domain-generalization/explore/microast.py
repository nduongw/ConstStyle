from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import explore.net_microAST as net
from dassl.utils import setup_logger, set_random_seed
from dassl.engine import build_trainer
from explore.sampler import InfiniteSamplerWrapper
from explore.util import *

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
args = get_args()
cfg = setup_cfg(args)

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

vgg = net.vgg
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

content_encoder = net.Encoder()
style_encoder = net.Encoder()
modulator = net.Modulator()
decoder = net.Decoder()

network = net.Net(vgg, content_encoder, style_encoder, modulator, decoder)
network.train()
network.to(device)

trainer = build_trainer(cfg, args)
train_loader_x = trainer.train_loader_x
test_loader = trainer.test_loader

content_iter = iter(cycle(train_loader_x))
style_iter = iter(cycle(test_loader))

optimizer = torch.optim.Adam([
    {'params':network.content_encoder.parameters()}, 
    {'params':network.style_encoder.parameters()}, 
    {'params':network.modulator.parameters()},
    {'params':network.decoder.parameters()}
    ], lr=args.lr)

start_iter = -1

# continue training from the checkpoint
# if args.resume:
#     checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
#     network.load_state_dict(checkpoints['net'])
#     optimizer.load_state_dict(checkpoints['optimizer'])
#     start_iter = checkpoints['epoch']

# training
for i in tqdm(range(start_iter+1, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter)
    style_images = next(style_iter)
    content_images, _ = trainer.parse_batch_test(content_images)
    style_images, _ = trainer.parse_batch_test(style_images)
    
    stylized_results, loss_c, loss_s, loss_contrastive = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_contrastive = args.SSC_weight * loss_contrastive
    loss = loss_c + loss_s + loss_contrastive

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'\nContent loss: {loss_c.item()} | Style loss: {loss_s.item()} | Contrastive loss: {loss_contrastive.item()}')
    
    ############################################################################
    # save intermediate samples
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i + 1) % 500 == 0: 
        visualized_imgs = torch.cat([content_images, style_images, stylized_results])
        
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(visualized_imgs, str(output_name), nrow=args.batch_size)
        print('[%d/%d] loss_content:%.4f, loss_style:%.4f, loss_contrastive:%.4f' \
               % (i+1, args.max_iter, loss_c.item(), loss_s.item(), loss_contrastive.item()))    
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.content_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'content_encoder_iter_{:d}.pth.tar'.format(i + 1))
        
        state_dict = network.style_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'style_encoder_iter_{:d}.pth.tar'.format(i + 1))
        
        state_dict = network.modulator.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'modulator_iter_{:d}.pth.tar'.format(i + 1))

        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))

        checkpoints = {
            "net": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": i
        }

