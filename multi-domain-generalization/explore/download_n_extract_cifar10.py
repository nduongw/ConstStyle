import torchvision.datasets as datasets
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

for idx, ele in enumerate(cifar_trainset):
    path = f'./DATA/CIFAR-10/train/{ele[1]}'
    os.makedirs(path, exist_ok=True)
    im = ele[0]
    if idx < 10:
        im.save(f'{path}/00000{idx+1}.jpg')
    elif idx > 10 and idx + 1 < 100:
        im.save(f'{path}/0000{idx+1}.jpg')
    elif idx > 100 and idx + 1 < 1000:
        im.save(f'{path}/000{idx+1}.jpg')
    elif idx > 1000 and idx + 1 < 10000:
        im.save(f'{path}/00{idx+1}.jpg')
    elif idx > 10000:
        im.save(f'{path}/0{idx+1}.jpg') 
