#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Custom Diffusion authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# Downloaded from 
# https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py

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

from unet.unet import ImprovedBottleNeck, Unet, AttentionBottleNeck, BCELoss_class_weighted

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

from postprocess import postProcess, visualizeBoundary

# floodfill
import floodfill

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)

os.environ["USE_FLASH_ATTENTION"]="1"


# take mask of dimension # B, H, W
def getBoundaryMask(x: torch.Tensor) -> torch.Tensor:
    top = x[:, 1 : -1, 1: - 1] - x[:, 0 : -2, 1: - 1]
    bot = x[:, 1 : -1, 1: - 1] - x[:, 2 : , 1: - 1]
    left = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 0: - 2]
    right = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 2: ]
    return torch.logical_or(torch.logical_or(top, bot), torch.logical_or(left, right))

# take a mask of dimension H, W, then break it into sections
# use the following internal codes:

UNCLAIMED_CELL = 0
UNCLAIMED_BOUNDARY = 1
FILLED_CELL = 2
FILLED_BOUNDARY = 3
CLAIMED_CELL_OR_BOUNDARY = 4

def segmentBoundaryMask(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint8)
    masks = []
    unclaimed = getFirstUnclaimed(x)
    while unclaimed is not None:
        # flood empty spaces and adjescent boundary
        filled = floodfill.flood(x, unclaimed[0], unclaimed[1]) 
        # mask out the fill 
        mask = np.logical_or(x == FILLED_CELL, x == FILLED_BOUNDARY)
        x = np.ma.array(x, mask=mask,fill_value=CLAIMED_CELL_OR_BOUNDARY)
        x = np.array(x.filled())
        # exclude dust from masks
        if(filled > 10):
            masks.append(mask) 
        unclaimed = getFirstUnclaimed(x)
    return np.stack(masks)

def getFirstUnclaimed(x: np.ndarray) -> np.ndarray:
    i,j = np.nonzero(np.logical_or(x == UNCLAIMED_CELL, x == UNCLAIMED_BOUNDARY))
    if len(i) > 0:
        return np.array([i[0], j[0]])
    else:
        return None
    
# equivalent to the C code except it runs hundreds of times slower    
def floodFill(area: np.ndarray, y: int, x: int):
    # python cannot handle tail recursion so this has to be done in a loop
    filled = 0
    checked = 0
    targets = [(y, x, False)]
    while len(targets) > 0:
        target = targets.pop()
        checked += 1
        if not isInBounds(area.shape, target[0], target[1]):
            #print(f"oob: {target[0]}, {target[1]}")
            continue
        val = area[target[0], target[1]]
        if target[2] and val == UNCLAIMED_BOUNDARY:
            #print(f"crossing boundary: {target[0]}, {target[1]}")
            continue
        if val != UNCLAIMED_BOUNDARY and val != UNCLAIMED_CELL:
            #print(f"not available: {target[0]}, {target[1]}")
            continue
        #print(f"filling: {target[0]}, {target[1]}")
        if val == UNCLAIMED_CELL:  
            is_boundary = False
            area[target[0],target[1]] = FILLED_CELL
            filled += 1
        if val == UNCLAIMED_BOUNDARY:
            is_boundary = True
            area[target[0],target[1]] = FILLED_BOUNDARY
            filled += 1
        targets = targets + [(target[0]-1, target[1], is_boundary), (target[0]+1, target[1], is_boundary), (target[0], target[1] -1, is_boundary), (target[0], target[1]+1, is_boundary)]
    print(f"filled: {filled}, checked: {checked}")
    return filled

def isInBounds(shape: tuple[int, int], y, x):
    return 0 <= y and y < shape[0] and 0 <= x and x < shape[1]


def boundaryMaskToTensor(masks: np.ndarray) -> np.ndarray: 
    masks = masks.astype(np.uint8)
    for i in range(masks.shape[0]):
        masks[i, :, :] *= (i + 1)
    return np.sum(masks, axis = 0)

# code to test boundary segmentation
def testBoundarySegmentation(seg: torch.Tensor, path: str):
    boundary = getBoundaryMask(seg)
    boundary = boundary.squeeze(0)
    segments = segmentBoundaryMask(boundary.cpu().numpy()) 
    stack = boundaryMaskToTensor(segments).astype(np.uint8)
    # normalize to 255 and shape to match
    seg = seg.squeeze(0)
    seg = seg * 10
    seg = seg[1: -1, 1: -1]
    seg = seg.cpu().numpy()
    boundary = boundary.to(torch.uint8) * 255 
    boundary = boundary.cpu().numpy()
    stack = stack * (255 // (2 * segments.shape[0]))
    # stack along width
    output = np.concatenate((seg, boundary, stack),axis = 1) 
    # save to file
    image = im.fromarray(output,'L')
    image.save(path)

# hard code in google colab choices
choices = np.array([  556, 10897,  4261, 12117,   422,  2726, 12278,  5084, 11531,
        3044,  9177,  6183,  6356,  2539,   970,  7438,  9086,   847,
        4712,  2770, 11199,  3725,  5325,  9849, 11290, 10024,  5610,
       11478,  8908, 11484,   469,  3413,  1420,    54,  3972,  7304,
        8600, 11128,  8416,  9423,  3306, 11949, 11297,  1230, 11129,
        4784,  7507, 11037, 12123,  7458, 11924,  9292,  8411,  5824,
        1310,   852,  9957,  5293,  8985,  7163, 11069,  3272,  8283,
        5920, 10384,  7483,  8754, 11323,  5491, 10109,  7135,  9234,
       11294,  8208,  7292,  6050,  8090,  1959,  3200,  4783,    84,
        6522,  6890,  1639,  3541, 11658,  5162,  7957, 10684,  7004,
         286,  2986,  2208,  3647,  3905,    98,  2046,  4513, 11801,
        7252, 10578, 11512,  4596,   253,  1211,  5964,  6131,  9784,
        5756,  6727,  3931,  4435,  9815, 12200,  6778,  3521, 10503,
        4976,  5830,  4272,  9842,  7961,  8880, 12296,  5576,  3826,
        9760, 10765, 12392, 10187,  6238,  8599,  4239,  8384,   672,
        7100,  8825,  9190,  7754, 11744,  7511,  6436,  7379,  9886,
        9765,  9731,  2821,  3113,   974,   161,  5713,  8377,  6793,
       11534,  5679,  2252,  3384,  2231,  2698,  7326,   517, 10905,
       10516,  2830,  2098,  3552, 12372, 10658, 12403,  5565,  5032,
         478,  8764,  9695,   929,  3813,  4556, 11685,  5204,  5839,
        1644,  7559, 11501,  9310,  9746,  2328, 12405, 11131, 11493,
       10695, 11852,  4855,  1928,  8271,  1903,  5426,  6105,  8797,
        2431,  7850,   124, 12176,  7456, 10769,  4512,  8763,  8209,
        1676,  8004,  8970,  3620,  4942,  8990,  5235, 11593,   892,
       11163,  5844,  4251,  5922,  1394,  7789,  7002,  2790,  1790,
        8266,  4339,  4840,  9987,  8234,  9646,  5092, 12291, 10531,
        8664,   776,  3662,  8527,  2018,  5690,  3899,  6598,  3275,
         906, 12155, 12659,  9832, 10143,   841,   653,  1353,   486,
        5773,  4486,  9785,  6068,  3675,  2874,  9350,  3142,  6083,
        8575,  5889,  9943, 10772,  1500, 12091,  6531,  8619, 12467,
        6000, 11255,  4221,  6161,  7328,  6200,  5051,  2306,  6430,
        4013,  9222,  2884,  3704,  7223,  7643,  6879,  7592,  6410,
        8227, 12672,  6631,  4113,  6711,  1116,  2645,  1734,  5560,
        5597,  4907,  3503, 12038, 10799,  3720,  7059,  6904, 11752,
        2554,  9862,  4799,  1801,  4835, 11407,  7453,  4707,  7176,
        9228,  5027,  7222,  3580,  1097,  3267, 11775,  1576, 11442,
        1401,  1520,  5575, 10744,  4751,  8125,  5979,  5430,  1746,
        1857, 11067,  5208,  7640,  3030,  8236,  6327,  6804,  4925,
        3548,  6373, 11140,  2924,  1709,  5259,  6743, 10281,  7498,
        2998, 12341,  6399,  5404,  6977, 11552, 12073,  2163,  9082,
       12006,  4377,  2041,  4854,  8370,  5413, 10431,  2585,   499,
          51,  7273,   828,   643,  6012,  7219, 12039, 11574,  6170,
        8383,  1180,  3198,   186,  2280,   626,  8680,  9611,  1834,
        5304, 10060,  1016,  7448, 12213,  8811,  6383,  5966,  8247,
        1477, 12391,  2387,  5397,   475,  5284,  9916,  9702,  8262,
       10015,  9140,  1423, 10985, 11464,  9485,  7924,   677,  2900,
       12148,  3409,  2826, 12231,  9497, 11614,  7181,  9719,   612,
        1853, 11891,    64,   428, 10785,  7783, 12420, 10801,   172,
        2351,  9591,  9217, 10138,  1697,  5380,  2858,  2919,   344,
       10251, 10651,  8083,  1074,  7403,  7722,  1621,  7908, 12531,
       11871,  7911,  1264,   513,  9947,  9247,  3375,  8137, 10453,
         525, 11856,  4649,  1315,  6481,  9294,  8782, 10927,  6134,
        3592,   143,  5264,  3754,  5458, 11527,  3516,  5135, 12594,
        8605,  5410,  1139,   146,  6486,  8174,   385,  7009,  5698,
        1453,  1462,  7593, 10465, 12112,  3230,  1412,  5053, 11343,
       10094,  8561,    58, 11770,  7778,  4490,  1669,  5720,  7312,
        9313,  4306, 11175, 11330,  6602,  2101,  8713,  1527, 12500,
        8809,  5231,  6630, 12008,  1030,  1766, 11488,  1883,  3741,
        3758,  6362,  4354,  9508,  3235,  7587,  9119,  9773, 10746,
        8831,  3991,  8740,  8031,  2492,  9267, 11030,  5010, 11988,
        7875,  5845,  1050,  8636,  1402, 12639,  1434, 10122, 11156,
        1666, 12153,  2824,  5897,  6204,  9579, 11029,  9157,  8724,
        6266,  8145,  5588, 12670,  6865,    94,  6777,  1551,  2055,
        2012,  7393,  6247,  1140,   601,  7083,  8665,  3042,  1081,
       11359, 11184, 12494, 11970,  9400,  5348,  7132,  4886,  9623,
        2035, 11523, 11537,  1651,  8696, 11251,  6290,  1015,  6984,
        8727,  7479,  1143,  7653,  7390, 11459,  1589, 11057, 10357,
        6057,  8362,  7191,  2436,   423,  8848,  2127,  3469,   872,
        7622,  9229,  6417,  6662,  9691,  2438,  7863,  5895, 10226,
       11576,  6454,  5551,  3958,  9287,  5798, 10300, 11051,  8091,
       11445, 12299,  3581,  2840,  8274,  2207,  4579,  5236,  3952,
        8895,  4423,  6097,  4437,  4084,  2808,  7296, 12205,  1994,
        5772,  5238,  1688,   738,  3219,  3018,  7729,  8293,  4654,
        9479,  5595, 10080, 10369, 11693,  6219,  3907,  6484,  9992,
        1627,  8941,  9425,  7457,  7775,  7598, 12454,  7013,  9709,
        3882, 11860,  2904, 10931,  9697, 10217, 12598,  1200,  8351,
       11205,  4896,   629,  1293, 10101,  1131,  1232, 10832,  6349,
       11225,  4530,    85,   597,  3423,  5049, 11510,  6930,  7796,
        3026,  2767,  2873, 10791,  2203,  5985,   527,  6513, 10334,
        9625, 10626,  8938,  6127, 11957,  3862, 10062,  4782,  9758,
        8272,  6270,  7184,  5217,   648, 10102,  2123,  2368,   673,
        8622,   257,  6571,  1404,  5543,  1868, 12667,  3464,  1896,
        2014,  7285, 11210, 11141,  1751, 11241,  3778,  3881, 10042,
        2342,  1780,  4359,  7241,  5792,   810, 10742,  1282, 11689,
        9952, 10249,  9093, 12124,  2885,  3276,  3022,  6475,  5535,
        3342, 10581, 11742,  7953,  1387, 12426, 11305,  4566,  1451,
        5035,  3538,  7362,  9197, 10595, 11395,  7400,  9344,  6482,
        4082,  2841,  5656,  9735,  3061, 10135,  3518, 12379,  8110,
         218, 10354,  9594,  7755,  5715,  5283,  9837,  9030, 11412,
        2219,  7291,  2030, 11822,    79,  5129,  9164,  1717,  8854,
        1296, 10293,  7762,  1814,  6798,  7310,   304,  6114, 11328,
       11764,  3509,  6384, 11710,  3844,  8124,  6231,  5989,  1620,
       12348, 11931,  4875,   991, 11375,  6957,  1517,  5321,  1607,
       10133,  9067,  1035,  8779,   308,  9869, 10609,  2452,  5119,
        7134,  9408,  8525, 11390,  6218,  3642,  1496, 11907,  8325,
         220,  1619,  3432,  2822,  2142,   171, 11486,  1769,  5332,
        6157,  6139, 11951, 12613,  9562, 12276, 12475,  3451,  2416,
         551, 10762,  8860,  7377,  4519,  2206,  9193,  1887, 10982,
       10234,   833,  1511,   393,  2968, 11675, 11541,    65,  3234,
        4682,  9203,  9078,  1164, 10925,  4606,  5057,  8158,  4138,
        9520,   272,  3896,  5791,  6400,  1430, 11023,  2756, 10992,
       10436,  9239,  1122, 11006,  1870, 12251, 11968,  5372,  3689,
         554,  6585,  2033,  2232, 11563, 12248,  3162, 11189,  6365,
        7685,  3595,  7847, 11779,  2506,  3921, 11910,  9994,  5689,
       10262,  3833,  4588,  6764,  6864, 12460,   462,  2926,  2849,
       12069, 11338,  7608,  1333,  3970,  9517, 11624, 12476,  9044,
        8167,  3020,  3483,  1472,  8329,  8252,  1345,  3912,  6866,
        3745,  1738,  6729, 11110,   122, 10246,  4509,   268,  3529,
        5444, 10541,  6459,  6761,  4114, 11835,  3313, 10023,  5876,
        6730,  7097,  1999, 11644,  5477,  6093,  8334,  6044,  3426,
        1027,   806,  9724,  5803,  5993,  5196, 11912,  1882,   283,
        3664,  7346,  7614,  4312,  5558,  7226, 12308, 11506,  3731,
        2122,  9871,  7311,  2179,  1286,  8434,  1798,  8222,  4106,
        2343,  6150,  2234,   518,  9799,  5178,  9434,  1051,  7832,
        8095,  1179,  7277,  1747,  7817,  3846,  5385,  8268,  8968,
        6143,  9457,   243, 10681, 12587,  2204,  6821,  7225, 10088,
        3817,  9003,  2525,  8785,  5688,  1592, 11266,  5580,  1203,
         989,  3399,  9201,  5611,  9853,  8953,  3261,  9728,  3292,
       12108,  9549,  3322,  5716,  8744,  6187, 11357,  4578, 12025,
        4237,  1475,  4284,  1096, 12002,  4525,  7595,  7884,  8553,
        3772,  2987,  4768,  2080,  5584,  6203,  8716,  6226,  1906,
       11745,  6274, 11278,  2019,  8981,  5785,  8728,  9049,  9666,
        1128,  1330,   284,  4189,  2092,  4026, 10204,  5239,  5095,
       12557,  8330, 12492,  9831,  6334,  6987,  1584,  8304,  5449,
        3684, 11515,  3531, 10739,   731, 10314,  2936,   435, 10715,
       12464,  5130,  3473,  2695, 12344,  5486,  4365,   387,  7574,
        3819,  9989,  7028,  1138,  6958,  7107,  7979,  7224,  9619,
         995,  7108, 10754,  5925,  8232,  8408,  5530,  1441,  3088,
        1759,   468,  7158,  6364,  5776,  6995, 12699,  3489,  8879,
        5938,  8373,  9828,  9912, 10037,    92, 10140,  2518, 10493,
        1761,  8025,  8602,  3019,  3634, 11322,  4885, 10612, 10148,
        5488, 10496,  9684,  3940,  1226,  8834,  7945, 10506,  2027,
        5516, 11867,  8454,  7125,  2864,  4168,  6006,  8668,  1640,
        5846,  2393, 10050,  5931,   964,  4680, 11441,  4117,  5191,
       12227,  4633,  9413, 11661,  8635,  9398,  1950,  6911,   553,
        6295,  5903,  4603,  2462,  9613,  9848,  8577, 11301,  5640,
       12285,   859,  4487,  8867,  9597,  2091,  7998,  3504, 10481,
       10870,   808, 11246,  7025,  1167,  2879,  5707,  5618,  6912,
       10427,  5100,  8161,  9907, 11749,  2314,  4701,  6344,  5711,
       10650,  1624,  6510,  9972,  9102,  9657,  9335, 12380,  6681,
        1195,  2557,  8852,  1091, 10900,  5497,  5572,  8312,  9829,
        1133])

def parse_args(input_args=None):

    ## Parameters to configure training and testing schedules
    parser = argparse.ArgumentParser(description="Custom Segmentation Training Script")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        default=None,
        help="Path to model parameters",
    )
    parser.add_argument(
        "--score_file_name",
        type=str,
        default='scores.csv',
        help="Name of csv file containing score information for distribution",
    )
    parser.add_argument(
        "--prediction_file_name",
        type=str,
        default='predictions.csv',
        help="Name of csv file containing score information for predictions",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        default=None,
        help="A folder containing raw images",
    )
    parser.add_argument(
        "--label_data_dir",
        type=str,
        required=True,
        default=None,
        help="A folder containing label information for some images",
    )
    # big daddy test sets since we ruined it anyway
    parser.add_argument(
        "--testing_proportion",
        type=float,
        default=0.01,
        help="Proportion of samples to be used for testing",
    )
    parser.add_argument(
        "--freeze_testing_rng",
        type=int,
        default= 42,
        help=("Use this rng seed each time to generate consistent test samples",
              "   set to -1 to generate new test samples each validation run")
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    ## parameters to determine dataset transformations ##
    parser.add_argument(
        "--fit_width",
        action="store_true",
        default=False,
        help=(
            "Always include maximum width in the crop"
        ),
    )
    parser.add_argument(
        "--max_scale_factor",
        type=float,
        default=1.0,
        help=(
            "Maximum amount source image is scaled in either direction before cropping",
            "Ignored if --fit_width is on"
        ),
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        default=False,
        help=(
            "Randomly flip images horizontally"
        ),
    )
    parser.add_argument(
        "--vertical_flip",
        action="store_true",
        default=False,
        help=(
            "Randomly flip images vertically, not recommended"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results/",
        required=True,
        help="Output Directory",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'Only Supported Platform is Tensorboard'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--model_features",
        type=int,
        default=48,
        help="base model feature size"
    )
    parser.add_argument(
        "--model_width",
        type=int,
        default=512,
        help="expected input width in pixels for model"
    )
    parser.add_argument(
        "--model_height",
        type=int,
        default=512,
        help="expected input height in pixels for model"
    )
    parser.add_argument(
        "--model_bottleneck",
        type=str,
        default="default",
        help="specify model bottleneck, either default, improved, attention"
    )
    parser.add_argument(
        "--model_use_local_attention",
        action="store_true",
        default=False,
        help="up blocks will use local attention with convolution"
    )
    parser.add_argument(
        "--model_norm",
        type=str,
        default="GroupNorm",
        help="use GroupNorm or GroupVectorNorm"
    )
    ## arguments for reverse process
    parser.add_argument(
        "--reverse_index",
        type=int,
        default=-1,
        help=(
            "Image in test set to perform reverse itteration on"
        ),
    )

    parser.add_argument(
        "--reverse_steps",
        type=int,
        default=100,
        help=(
            "Number of itterations when running in reverse"
        ),
    )
    parser.add_argument(
        "--reverse_log_frequency",
        type=int,
        default=10,
        help=(
            "How often intermediate steps are saved"
        ),
    )
    parser.add_argument(
        "--reverse_lr",
        type=float,
        default=1e-3,
        help=(
            "Learning Rate Reverse Process"
        ),
    )
    parser.add_argument(
        "--reverse_norm_scale",
        type=float,
        default=1.0/ (512* 3**0.5),
        help=(
            "Higher values prioritize smaller changes over increased accuracy"
        ),
    )
    parser.add_argument(
        "--reverse_path",
        type=str,
        default='./reverse/',
        help=(
            "Folder to hold reverse process progression"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def class_to_one_hot(input: torch.Tensor, num_classes = -1, dim = -1):
    output = torch.nn.functional.one_hot(input.to(torch.int64).unsqueeze(dim = dim), num_classes).to(dtype=torch.float32)
    output = output.transpose(dim, -1)
    output = torch.squeeze(output, -1)
    return output

# compute intersection and union for all categories assuming shape B,C,H,W
# return statistics 4, B, C first dimension is intersection, union, a_mass, b_mass for batch and channel
def iou(a: torch.Tensor, b: torch.Tensor): 
    intersection = torch.sum(a * b, dim=(2,3))
    union = torch.sum(a + b - a*b, dim=(2,3)) 
    a_mass = torch.sum(a, dim=(2,3))
    b_mass = torch.sum(b, dim=(2,3))
    iou = (intersection + 1)/(union + 1) 
    return torch.stack((intersection, union, a_mass, b_mass, iou), dim = 0) 

#hack
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CLASSES = len(labels)

def logImageSet(path: str, src: np.ndarray, target: np.ndarray, guess: np.ndarray):
    # create a color map
    cmap1 = colormaps.tab20
    colors1 = cmap1(np.linspace(0.025,1.025,20))
    colors = np.concatenate((np.array([[0.0, 0.0,0.0, 1.0]]), np.array(colors1), np.array([[240.0/255, 2.0/255, 127.0/255, 1.0],[1.0, 1.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0]])))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_map', colors, N = 24)
    bounds = np.arange(0.5, CLASSES - 1, 1) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig = plt.figure(figsize=(512.0/196, 2048/196), dpi=196)
    columns = 4
    rows = 1
    src = np.transpose(src, [1,2,0])
    target = target.squeeze(0)
    guess =  guess.squeeze(0)

    # ax enables access to manipulate each of subplots
    ax = []

    # determine all labels that we need
    labels_used = sorted(np.unique(np.stack((target, guess))).tolist())

    ax.append( fig.add_subplot(rows, columns, 1))
    ax[-1].set_title('input')
    plt.imshow(im.fromarray(src))
    ax.append( fig.add_subplot(rows, columns, 2))
    ax[-1].set_title('target')
    plt.imshow(im.fromarray(target), cmap = cmap, norm=norm, interpolation='none')
    ax.append( fig.add_subplot(rows, columns, 3))
    ax[-1].set_title('guess')
    plt.imshow(im.fromarray(guess), cmap = cmap, norm=norm, interpolation='none')
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

# Create a dummy model to store the image as a parameter
class InputModel(torch.nn.Module): 
    def __init__(self, image: torch.Tensor):
        super().__init__()
        self.input = torch.nn.Parameter(image.clone(),requires_grad=True)
    def forward(self):
        return self.input.clamp(0,1.0)

def logReverseImageSet(path: str, src: np.ndarray, guesses: np.ndarray):
    # do something
    guesses = guesses.transpose((1,2,0,3))
    guesses = guesses.reshape((guesses.shape[0],guesses.shape[1], guesses.shape[2] * guesses.shape[3]))
    # concatenate along width
    output = np.concatenate((src, guesses), 2)
    output = output.transpose((1,2,0))
    # now save the image
    image = im.fromarray(output)
    image.save(path)
    return

# code to run network in reverse (train input to match target with model fixed)
def reverse(args, accelerator: Accelerator, unet: Unet, dataset: ImageSegmentationDataset):
    index = args.reverse_index
    rawInput, target = dataset[index]
    # prepare for later
    rawInput = torch.unsqueeze(torch.mul(rawInput.to(dtype=torch.float32, device = accelerator.device), 1/255.0),0)
    target = target.to(dtype = torch.long, device = accelerator.device)
    # initialize the model
    imageModel: InputModel = InputModel(rawInput) 
    optimizer = torch.optim.AdamW(
        imageModel.parameters(),
        lr=args.reverse_lr,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-8,
    )
    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.reverse_steps * accelerator.num_processes,
    )

    imageModel.to(device = accelerator.device)
    unet.to(device = accelerator.device) 
    imageModel, unet, optimizer, lr_scheduler = accelerator.prepare(imageModel, unet, optimizer, lr_scheduler) 
    imageModel.train()
    unet.train()
    bdry_loss_fn = BCELoss_class_weighted.apply
    target_bdry = getBoundaryMask(target).to(dtype=torch.float32)
    # store results here
    output = torch.zeros((math.floor(args.reverse_steps / args.reverse_log_frequency), 3, 512,512)).to(dtype = torch.float32, device = accelerator.device) 
    progress_bar = tqdm(
        range(0, args.reverse_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for i in range(args.reverse_steps):
        with accelerator.accumulate(imageModel):
            input = imageModel.forward()
            logits, bdry = unet(input)
            bdry = bdry[:,:, 1:-1, 1:-1].squeeze(1)
            if i == 0 or i == args.reverse_steps - 1:
                # log start and end points
                guess_np = torch.argmax(logits,1).to(torch.uint8).unsqueeze(1).cpu().numpy() 
                target_np = target.unsqueeze(0).to(torch.uint8).cpu().numpy()
                source_np = torch.mul(input, 255).to(torch.uint8).cpu().numpy()
                logImageSet(os.path.join(args.reverse_path, 'image_'+ str(index) + '_' + str(i)+'.png'), np.squeeze(source_np, 0), np.squeeze(target_np, 0), np.squeeze(guess_np, 0)) 
                # also log the boundary
                visualizeBoundary(bdry.squeeze(0).clone().detach().cpu().numpy(), target_bdry.squeeze(0).clone().detach().cpu().numpy(),os.path.join(args.reverse_path, 'bdry_image_'+ str(index) + '_' + str(i)+'.png'))
            
            # compute segmentation accuracy
            accuracy = torch.nn.functional.cross_entropy(logits,target,reduction='mean')  
            # compute boundary accuracy
            
            bdry_loss: torch.Tensor = bdry_loss_fn(bdry,target_bdry,[1.0, 5.0]).mean()
            drift = args.reverse_norm_scale * torch.functional.norm(input - rawInput, p = 2)
            loss = accuracy + bdry_loss + drift
            accelerator.backward(loss)
            progress_bar.set_postfix(**{'accuracy': accuracy.item(), 'drift': drift.item(), 'bdry': bdry_loss.item()})
            progress_bar.update(1)
            # check gradients
            if accelerator.sync_gradients:
                params_to_clip = imageModel.parameters()
                accelerator.clip_grad_norm_(params_to_clip, 1)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                if i % args.reverse_log_frequency == 0:
                    output[math.floor(i/ args.reverse_log_frequency), : , : ,:] = torch.squeeze(input, 0) 
    
    # log the results
    np_source = torch.squeeze(torch.mul(rawInput, 255),0).to(dtype=torch.uint8).cpu().numpy()    
    np_output = torch.mul(output, 255).to(dtype=torch.uint8).cpu().numpy()    

    logReverseImageSet(os.path.join(args.reverse_path, 'reverse_' + str(index)+'.png'), np_source, np_output)
    progress_bar.close()
    return


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        cpu=False,
        gradient_accumulation_steps=1,
        mixed_precision= args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom segmentation", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # load model with specified bottleneck
    unet = Unet(3, CLASSES, args.model_height, args.model_width, base_features=args.model_features, bottleneck = args.model_bottleneck, useAttention=args.model_use_local_attention, norm=args.model_norm)
    if args.pretrained_model_path is not None:
        unet.load_state_dict(torch.load(args.pretrained_model_path)) 
    else:
        raise ValueError("must specify a pretrained model for testing")

    # For mixed precision training 
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # unet = unet.to(accelerator.device, dtype=weight_dtype)
    unet = unet.to(accelerator.device)          # try without cast

    # if we are upgrading the bottleneck, only the bottleneck gets trained
    # that way we can bring the model to where it was before (roughly) and then train it from there
    # this doesn't really matter for the convolution model which trains very fast but it does matter 
    # for the attention model that trains very slowly

    # params = unet.bottleneck.parameters()
    accelerator.register_for_checkpointing(unet)


    train_dataset = ImageSegmentationDataset(args.raw_data_dir, args.label_data_dir, width=args.model_width, height=args.model_height, transform= 'FitWidth' if args.fit_width else 'unsupported', seed = args.seed)
    test_dataset = train_dataset.split(args.testing_proportion, choices=choices)

    # not going to use a data loader for basic testing
    
    # prepare unet in case it has to be split or something which is very unlikely for pure testing
    unet = accelerator.prepare(unet)

    if args.reverse_index >= 0:
        # we are doing a different test
        print("Performing Reverse Process:") 
        reverse(args, accelerator, unet, test_dataset)
        exit(0)

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")

    progress_bar = tqdm(
        range(0, len(test_dataset)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # create an array to store all the stats
    output = np.empty((len(test_dataset) + 1, 5 * CLASSES + 4),dtype=object)

    stat_header = np.empty((5,CLASSES),dtype=object)

    for x in range(CLASSES):
        stat_header[0, x] = labels[x] + ' intersection'
        stat_header[1, x] = labels[x] + ' union'
        stat_header[2, x] = labels[x] + ' predicted mass'
        stat_header[3, x] = labels[x] + ' true mass'
        stat_header[4, x] = labels[x] + ' iou score'

    output[0, 0:2] = ['index','file']
    output[0, 2: 2 + 5 * CLASSES] = np.reshape(stat_header, 5 * CLASSES)
    output[0, 2 + 5 * CLASSES : ] = ['pixel accuracy', 'model pixel accuracy'] 

    output_predictions = output.copy()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    imagePath = os.path.join(args.output_dir, 'images')
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)

    

    # run a test on a portion of the training data to compare
    test_dataset = train_dataset.split(percentage = 0.1)

    with torch.no_grad():
        unet.eval()
        for i in range(len(test_dataset)):
            input,target = test_dataset[i]
            raw_input = input
            # unsqueeze a batch dimension of 1
            input = torch.unsqueeze(input.to(weight_dtype)/255, 0).to(device=accelerator.device)        # 1, 3, H, W
            target = torch.unsqueeze(target, 0).to(device=accelerator.device)                           # 1, 1, H, W
            # compute logits and boundary information
            logits, boundary = unet(input)                                                              # (1, 24, H, W), (1, 1, H, W)
            logits = torch.nn.functional.softmax(logits, 1)                                             # (1, 24, H, W)
            # convert target to onehot
            target_np = target[:,:,1:-1,1:-1].to(torch.uint8).cpu().numpy()                             # (1, 1, H - 2, W -2)
            source_np = torch.unsqueeze(raw_input, 0).to(torch.uint8).cpu().numpy()                     # (1, 3, H, W)
            target = class_to_one_hot(target.squeeze(1), CLASSES, 1)                                    # (1, 24, H, W)
            # postprocess logits and boundary into guess
            guess_np = postProcess(logits.squeeze(0).cpu().numpy(), boundary.squeeze(0).squeeze(0).cpu().numpy())   # (H - 2, W - 2)
            # put back on the gpu, conver to one hot vectors
            predicted_logits = torch.Tensor(guess_np).cuda().unsqueeze(0)                               # (1, H-2, W -2)              
            predicted_logits = class_to_one_hot(predicted_logits, CLASSES, 1)                           # (1, 24, H-2, W-2)
            # trim target
            target = target[:,:,1:-1,1:-1]                                                              # (1, 24, H-2, W-2)
            wh = target.shape[2] * target.shape[3]
            prediction_stats = iou(predicted_logits, target).cpu().numpy()


            output_predictions[i + 1, 0] = i
            output_predictions[i + 1, 1] = test_dataset.seg[i] 
            output_predictions[i + 1, 2: 2 + 5 * CLASSES] = np.reshape(prediction_stats, 5 * CLASSES)
            output_predictions[i + 1, 2 + 5 * CLASSES : ] = np.array([np.sum(prediction_stats[0,0, :])/ wh, np.sum(prediction_stats[0,0, 1:])/ (wh - prediction_stats[3,0,0])])

            # pass (3, H, W), (1, H-2, W-2), (1, H-2, W-2)
            logImageSet(os.path.join(imagePath, 'image_' + str(i)+'.png'), np.squeeze(source_np, 0), np.squeeze(target_np, 0), guess_np[np.newaxis, ...]) 

            # output to tensorboard, if you prefer it
            """
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_images("guess", guess_np * 10, i, dataformats="NCHW")
                    tracker.writer.add_images("target", target_np * 10, i, dataformats="NCHW")
                    tracker.writer.add_images("source", source_np, i, dataformats="NCHW")
            """
            progress_bar.update(1)
    
    # df = pd.DataFrame(data = output[1:,1:], index = output[1:, 0], columns = output[0, 1:])
    # df.to_csv(os.path.join(args.output_dir, args.score_file_name))

    df_pred = pd.DataFrame(data = output_predictions[1:,1:], index = output_predictions[1:, 0], columns = output_predictions[0, 1:])
    df_pred.to_csv(os.path.join(args.output_dir, args.prediction_file_name))

if __name__ == "__main__":
    args = parse_args()
    main(args)