# Portions removed by Dmitri Chouchkov 
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import diffusers

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from tqdm.auto import tqdm
from dataset import ImageSegmentationDataset, labels 

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from unet.unet import ImprovedBottleNeck, Unet, AttentionBottleNeck

from self_attention_cv.bottleneck_transformer import BottleneckModule

# just for saving a csv file
import pandas as pd

# for plotting output
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.cm as colormaps
from PIL import Image as im

from postprocess import postProcess

# floodfill
import floodfill

CLASSES = len(labels)


def logImageSet(path: str, src: np.ndarray, raw:np.ndarray, guess: np.ndarray):
    # create a color map
    cmap1 = colormaps.tab20
    colors1 = cmap1(np.linspace(0.025,1.025,20))
    colors = np.concatenate((np.array([[0.0, 0.0,0.0, 1.0]]), np.array(colors1), np.array([[240.0/255, 2.0/255, 127.0/255, 1.0],[1.0, 1.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0]])))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_map', colors, N = 24)
    bounds = np.arange(0.5, CLASSES - 1, 1) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig = plt.figure(figsize=(src.shape[0]/196, src.shape[0] * 4/196), dpi=196)
    columns = 4
    rows = 1
    # ax enables access to manipulate each of subplots
    ax = []
    # determine all labels that we need
    labels_used = sorted(np.unique(np.stack((guess, raw))).tolist())

    ax.append( fig.add_subplot(rows, columns, 1))
    ax[-1].set_title('input')
    plt.imshow(im.fromarray(src))
    
    ax.append( fig.add_subplot(rows, columns, 2))
    ax[-1].set_title('raw')
    plt.imshow(im.fromarray(raw,'L'), cmap = cmap, norm=norm, interpolation='none')

    ax.append( fig.add_subplot(rows, columns, 3))
    ax[-1].set_title('processed')
    plt.imshow(im.fromarray(guess,'L'), cmap = cmap, norm=norm, interpolation='none')

    ax.append(fig.add_subplot(rows, columns, 4))
    # add a legend or something
    handles = [
        Rectangle((0,0),1,1, color=cmap(norm(key))) for key in labels_used
    ]
    label_list = [labels[v] for v in labels_used]
    ax[-1].legend(handles, label_list, loc='upper left', mode='expand', fancybox=True, ncol=1)
    ax[-1].figure.set_figwidth(15)
    ax[-1].figure.set_figheight(5)
    plt.axis('off')
    # ax[-1].figure.axes.get_xaxis().set_visible(False)
    # ax[-1].figure.axes.get_yaxis().set_visible(False)
    plt.savefig(path,dpi=196) 
    plt.close()

def main():
    unet = Unet(3, CLASSES, 512, 512, base_features=48, bottleneck = 'attention', useAttention=False, norm='GroupNorm')
    unet.load_state_dict(torch.load('./unet/output.ft.params'))
    unet.cuda() 
    test_dataset = ImageSegmentationDataset('../googleimg/', None, width=512, height=512, transform= 'FitWidth', seed = 42)
    output_dir = './google_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imagePath = os.path.join(output_dir, 'images')
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)
    
    weight_dtype = torch.float32
    print(len(test_dataset))
    with torch.no_grad():
        unet.eval()
        for i in range(len(test_dataset)):
            print(i)
            input,target = test_dataset[i]
            raw_input = input
            # unsqueeze a batch dimension of 1
            input = torch.unsqueeze(input.to(weight_dtype)/255, 0).cuda()                               # 1, 3, H, W
            # compute logits and boundary information
            logits, boundary = unet(input)                                                              # (1, 24, H, W), (1, 1, H, W)
            logits = torch.nn.functional.softmax(logits, 1)                                             # (1, 24, H, W)
            # convert target to onehot
            source_np = torch.unsqueeze(raw_input, 0).to(torch.uint8).cpu().numpy()                     # (1, 3, H, W)

            raw_np = torch.argmax(logits,1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()[1:-1,1:-1]         # (H-2, W-2)
            # postprocess logits and boundary into guess
            guess_np = postProcess(logits.squeeze(0).cpu().numpy(), boundary.squeeze(0).squeeze(0).cpu().numpy())   # (H - 2, W - 2)

            # pass (3, H, W), (1, H-2, W-2), (1, H-2, W-2)
            logImageSet(os.path.join(imagePath, 'image_' + str(i)+'.png'), np.squeeze(source_np, 0).transpose([1,2,0]), raw_np, guess_np) 

if __name__ == "__main__":
    main()
    