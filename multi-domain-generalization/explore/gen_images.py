import net
from pathlib import Path
import torch.nn as nn
import torch
import os
# import wandb
import numpy as np
from dassl.utils import setup_logger, set_random_seed
from dassl.engine import build_trainer
from PIL import Image
from torchvision.utils import save_image
from explore.util import *

do_interpolation = False

def get_style_image():
    classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    rand_idx = np.random.randint(10, 20)
    rand_cls = np.random.randint(0, 7)

    if args.selected_domain == 'photo':
        if rand_cls == 0:
            style_path = f'{args.style}{classes[rand_cls]}/056_00{rand_idx}.jpg'
        elif rand_cls == 1:
            style_path = f'{args.style}{classes[rand_cls]}/064_00{rand_idx}.jpg'
        elif rand_cls == 2:
            style_path = f'{args.style}{classes[rand_cls]}/084_00{rand_idx}.jpg'
        elif rand_cls == 3:
            style_path = f'{args.style}{classes[rand_cls]}/063_00{rand_idx}.jpg'
        elif rand_cls == 4:
            style_path = f'{args.style}{classes[rand_cls]}/105_00{rand_idx}.jpg'
        elif rand_cls == 5:
            style_path = f'{args.style}{classes[rand_cls]}/pic_00{rand_idx}.jpg'
        elif rand_cls == 6:
            style_path = f'{args.style}{classes[rand_cls]}/253_00{rand_idx}.jpg'
    elif args.selected_domain == 'art_painting':
        style_path = f'{args.style}{classes[rand_cls]}/pic_0{rand_idx}.jpg'
    
    return style_path

args = get_args()
print('Done get arguments\n')

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

output_dir = Path(args.store_folder)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
    
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
    
trainer = build_trainer(cfg, args)

device = torch.device('cuda:0')
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load('experiments2/decoder_iter_96000.pth.tar'))
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

train_loader_x = trainer.train_loader_x
test_loader = trainer.test_loader

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

print(f'Content dir: {args.store_folder}')

for content_path in content_paths:
    string_content_path = str(content_path)
    extended_str = string_content_path.split('/')[-1].split('.')[-1]
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights, device)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        style_path = get_style_image()
        print(f'Style image: {style_path}')
        style_path = Path(style_path)
        while not os.path.exists(style_path):
            style_path = get_style_image()
            print(f'Another Style image: {style_path}')
        
        content = content_tf(Image.open(str(content_path)))
        style = style_tf(Image.open(str(style_path)))
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        output = output.cpu()

        output_name = output_dir / '{:s}.{:s}'.format(
            content_path.stem, extended_str)
        save_image(output, str(output_name))

print('Done')
