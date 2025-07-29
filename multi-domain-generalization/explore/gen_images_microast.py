from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import explore.net_microAST as net
from dassl.utils import setup_logger, set_random_seed
from dassl.engine import build_trainer
from explore.sampler import InfiniteSamplerWrapper
from explore.util import *
import traceback

def get_style_image():
    dataset = trainer.test_loader.dataset
    random_index = int(np.random.random() * len(dataset))
    random_sample = dataset[random_index]
    style_path = random_sample['impath']
    return style_path

args = get_args()
cfg = setup_cfg(args)

if cfg.SEED >= 0:
    print('Setting fixed seed: {}'.format(cfg.SEED))
    set_random_seed(cfg.SEED)
setup_logger(cfg.OUTPUT_DIR)

if torch.cuda.is_available() and cfg.USE_CUDA:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

output_dir = Path(args.store_folder)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

trainer = build_trainer(cfg, args)
device = torch.device('cuda')

content_encoder = net.Encoder()
style_encoder = net.Encoder()
modulator = net.Modulator()
decoder = net.Decoder()

content_encoder.eval()
style_encoder.eval()
modulator.eval()
decoder.eval()

content_encoder.load_state_dict(torch.load(args.content_encoder))
style_encoder.load_state_dict(torch.load(args.style_encoder))
modulator.load_state_dict(torch.load(args.modulator))
decoder.load_state_dict(torch.load(args.decoder))

network = net.TestNet(content_encoder, style_encoder, modulator, decoder)

network.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    string_content_path = str(content_path)
    extended_str = string_content_path.split('/')[-1].split('.')[-1]
    for i in range(5):
        style_path = get_style_image()
        print(f'Style image: {style_path}')
        style_path = Path(style_path)
        while not os.path.exists(style_path):
            style_path = get_style_image()
            print(f'Another Style image: {style_path}')
                
        try:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            with torch.no_grad():
                output = network(content, style, args.alpha)
                
            output = output.cpu()

            output_name = output_dir / '{:s}_{:d}.{:s}'.format(
            content_path.stem, i, extended_str)
            save_image(output, str(output_name))    
        except:
            traceback.print_exc()