import numpy as np
import floodfill
from PIL import Image as im
from scipy.signal import convolve2d
from floodfill import first_unclaimed
import faulthandler

import torch

# for plotting output
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.cm as colormaps

# for timing purposes
import time

# convert targets to boundary
def getBoundaryMask(x: np.ndarray) -> np.ndarray:
    top = x[:, 1 : -1, 1: - 1] - x[:, 0 : -2, 1: - 1]
    bot = x[:, 1 : -1, 1: - 1] - x[:, 2 : , 1: - 1]
    left = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 0: - 2]
    right = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 2: ]
    return np.logical_or(np.logical_or(top, bot), np.logical_or(left, right))

# necessary constants
UNCLAIMED_CELL = 0
UNCLAIMED_BOUNDARY = 1
FILLED_CELL = 2
FILLED_BOUNDARY = 3
CLAIMED_CELL_OR_BOUNDARY = 4

# code to segment boundary
def segmentBoundaryMask(x: np.ndarray) -> np.ndarray:
    faulthandler.enable()
    x = x.astype(np.uint8)
    masks = []
    unclaimed = getFirstUnclaimed(x, last_index = -1)
    while unclaimed is not None:
        # flood empty spaces and adjescent boundary, populate both filled and mask
        mask = np.zeros_like(x)
        filled = floodfill.flood(x, mask, unclaimed[0], unclaimed[1]) 
        # exclude dust from masks
        if(filled > 10):
            masks.append(mask) 
        unclaimed = getFirstUnclaimed(x, last_index= unclaimed[0] * x.shape[0] + unclaimed[1]) 
    return np.stack(masks)

def getFirstUnclaimed(x: np.ndarray, last_index = -1) -> np.ndarray:
    index = first_unclaimed(x, last_index)
    if index >= 0:
        j = index % x.shape[1]
        i = (index - j) // x.shape[0]
        return [i,j]
    else:
        return None

def expand(x: np.ndarray, itterations = 1) ->np.ndarray:
    kernel = np.ones([ 2* itterations + 1, 2 * itterations + 1],dtype=np.uint8)
    return (convolve2d(x.astype(np.uint8),kernel,mode='full') >= 1).astype(np.uint8)

def contract(x: np.ndarray,itterations = 1) -> np.ndarray:
    size = 2 * itterations + 1
    kernel = np.ones([ size, size],dtype=np.uint8)
    for i in range(itterations):
        for j in range(itterations):
            if i < itterations - j:
                kernel[i, j] = 0
                kernel[size -i - 1, j] = 0
                kernel[i, size -j -1] = 0
                kernel[size -i - 1, size -j -1] = 0
    return (convolve2d(1 - x.astype(np.uint8),kernel,mode='valid') == 0).astype(np.uint8)

def border(x:np.ndarray) -> np.ndarray:
    y = x.copy()
    y[0, :] = 1.0
    y[-1, :] = 1.0
    y[:, 0] = 1.0
    y[:, -1] = 1.0
    return y

def hollow(x: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype=np.uint8) 
    interior = convolve2d(x, kernel,mode='same',boundary='fill', fillvalue=1)
    output = x.copy()
    output[interior == 12] = 0
    return output

# compute statistics for each mask
# masks: N,510,510
# seg: 510,510,24
# this operation is expensive and can be sped up on the GPU
def getMaskStats(masks: np.ndarray, seg: np.ndarray, useGPU = True):
    if(useGPU):
        return torch.sum(torch.Tensor(masks).cuda().unsqueeze(-1) * torch.Tensor(seg).cuda().unsqueeze(0),dim= (1,2)).cpu().numpy()
    return np.sum(masks[..., np.newaxis] * seg[np.newaxis, ...], axis=(1,2))

    
# replace each mask with most likely block

def poolSegmentation(masks: np.ndarray, seg: np.ndarray, stats: np.ndarray):
    argmax = np.argmax(stats,axis=1)
    return np.sum(masks * argmax[:, np.newaxis, np.newaxis], axis= 0).astype(np.uint8) 

# output masks

#hack
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CLASSES = 24

labels = {
    0: 'background',	
    1: 'top',	
    2: 'outer',	
    3: 'skirt',
    4: 'dress',	
    5: 'pants',	
    6: 'leggings',	
    7: 'headwear',
    8: 'eyeglass',	
    9: 'neckwear',	
    10: 'belt',	
    11: 'footwear',
    12: 'bag',	
    13: 'hair',	
    14: 'face',	
    15: 'skin',
    16: 'ring',	
    17: 'wrist wearing',	
    18: 'socks',	
    19: 'gloves',
    20: 'necklace',	
    21: 'rompers',	
    22: 'earrings',	
    23: 'tie'
}

def logImageSet(path: str, src: np.ndarray, target: np.ndarray, raw:np.ndarray, partial:np.ndarray, guess: np.ndarray):
    # create a color map
    cmap1 = colormaps.tab20
    colors1 = cmap1(np.linspace(0.025,1.025,20))
    colors = np.concatenate((np.array([[0.0, 0.0,0.0, 1.0]]), np.array(colors1), np.array([[240.0/255, 2.0/255, 127.0/255, 1.0],[1.0, 1.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0]])))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_map', colors, N = 24)
    bounds = np.arange(0.5, CLASSES - 1, 1) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig = plt.figure(figsize=(src.shape[0]/196, src.shape[0] * 6/196), dpi=196)
    columns = 6
    rows = 1
    # src = np.transpose(src, [1,2,0])
    # target = target.squeeze(0)
    # guess =  guess.squeeze(0)

    # ax enables access to manipulate each of subplots
    ax = []

    # determine all labels that we need
    labels_used = sorted(np.unique(np.stack((target, raw))).tolist())

    ax.append( fig.add_subplot(rows, columns, 1))
    ax[-1].set_title('input')
    plt.imshow(im.fromarray(src))
    
    ax.append( fig.add_subplot(rows, columns, 2))
    ax[-1].set_title('target')
    plt.imshow(im.fromarray(target, 'L'), cmap = cmap, norm=norm, interpolation='none')
    
    ax.append( fig.add_subplot(rows, columns, 3))
    ax[-1].set_title('raw')
    plt.imshow(im.fromarray(raw,'L'), cmap = cmap, norm=norm, interpolation='none')

    ax.append( fig.add_subplot(rows, columns, 4))
    ax[-1].set_title('partial')
    plt.imshow(im.fromarray(partial,'L'), cmap = cmap, norm=norm, interpolation='none')

    ax.append( fig.add_subplot(rows, columns, 5))
    ax[-1].set_title('processed')
    plt.imshow(im.fromarray(guess,'L'), cmap = cmap, norm=norm, interpolation='none')

    ax.append(fig.add_subplot(rows, columns, 6))
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

# code to visualize boundary operations
# boundary: H,W <float32>
# truth: H, W <uint8>
def visualizeBoundary(boundary: np.ndarray, truth: np.ndarray, path: str, boundary_threshold: float = 0.1, expand_pixel_size: int = 5):
    # convert boundary to an inverted greyscale 3 channel image
    b_img = 1.0 - boundary
    b_img = np.clip(b_img,a_min=0.0, a_max=1.0)
    b_img = np.stack((b_img, b_img, b_img), axis= -1)
    # now do our post processing
    f = (boundary > boundary_threshold).astype(np.uint8)
    t = truth
    tl = convolve2d(f,[[1,1,0],[1,1,0],[0,0,0]],mode='same')
    tr = convolve2d(f,[[0,1,1],[0,1,1],[0,0,0]],mode='same')
    bl = convolve2d(f,[[0,0,0],[1,1,0],[1,1,0]],mode='same')
    br = convolve2d(f,[[0,0,0],[0,1,1],[0,1,1]],mode='same')
    in_block = np.any([tl == 4, tr == 4, bl == 4, br == 4],axis=0).astype(np.uint8)

    in_block = border(in_block)
    filled_block = expand(in_block, 5)
    filled_block = contract(filled_block, 5)

    in_block = np.logical_or(hollow(filled_block), in_block).astype(np.uint8)
    correct = np.logical_and(t,in_block)
    missing = t - correct
    extra = in_block - correct
    blank = 1 - np.logical_or(t,in_block)

    processed = apply_mask_color(correct.astype(np.uint8),[0, 255, 0]) + \
            apply_mask_color(missing.astype(np.uint8),[255, 0, 0]) + \
            apply_mask_color(extra.astype(np.uint8),[100, 100, 100]) + \
            apply_mask_color(blank.astype(np.uint8),[255, 255, 255])


    # draw the image
    b_img = (b_img * 255).astype(np.uint8)
    image = im.fromarray(np.concatenate((b_img, processed), axis =1))
    image.save(path)

def postProcess(seg: np.ndarray, 
                boundary: np.ndarray, 
                boundary_threshold: float = 0.1, 
                expand_pixel_size: int = 5,
                promote_pixel_threshold: float = 0.5,
                join_threshold: float = 0.05,
                impatience: float = 1.5,
                return_partial = False) -> np.ndarray:
    # snap boundary 

    tic = time.perf_counter()

    f_boundary = (boundary > boundary_threshold).astype(np.uint8)  
    # the boundary is uncertain and must be removed 
    seg = seg[ :, 1:-1,1:-1]
    f = f_boundary[1:-1,1:-1]
    # ignore boundary pixels not part of a 2x2 block
    tl = convolve2d(f,[[1,1,0],[1,1,0],[0,0,0]],mode='same')
    tr = convolve2d(f,[[0,1,1],[0,1,1],[0,0,0]],mode='same')
    bl = convolve2d(f,[[0,0,0],[1,1,0],[1,1,0]],mode='same')
    br = convolve2d(f,[[0,0,0],[0,1,1],[0,1,1]],mode='same')
    in_block = np.any([tl == 4, tr == 4, bl == 4, br == 4],axis=0).astype(np.uint8)
    # add a border so that we can close border gaps
    in_block = border(in_block)
    # perform a single expand contract step to try and close small gaps
    filled_block = expand(in_block, expand_pixel_size)
    filled_block = contract(filled_block, expand_pixel_size)
    # hollow out convex regions created by the expansion then combine block pixels to get final boundary
    in_block = np.logical_or(hollow(filled_block), in_block).astype(np.uint8)
    # use floodfill to segment boundary regions into masks
    masks = segmentBoundaryMask(in_block)
    # accumulate segmentation category scores for each masked region
    old_shape = seg.shape
    seg = np.ascontiguousarray(seg.transpose([1,2,0]), dtype=np.float32)        # 0.005 sec
    stats = getMaskStats(masks, seg)                                            # 0.2 - 6 sec
    # add a new mask for each label with point mass stats
    stats = np.concatenate((stats, np.eye(seg.shape[2], seg.shape[2], dtype=np.float32)), axis= 0)
    masks = np.concatenate((masks, np.zeros(old_shape)), axis = 0)              # 0.08 sec

    partial = poolSegmentation(masks, seg, stats) 

    # populate mappings, set unmapped to -1
    mapping = np.sum(masks * np.array(range(1, masks.shape[0] + 1))[..., np.newaxis, np.newaxis], axis = 0) - 1 # 0.15 sec
    mapping = mapping.astype(np.intp)
    masks = masks.astype(np.uint8)
    # for each unmapped pixel:
    #   if its top category probability exceeds promotion threshold, promote the pixel to the mask for that label
    #   if the pixel is not orthogonally adjescent to any mask, wait until it is
    #   if the pixel is orthogonally adjescent to a mask, see if the pixel statistics are within join_threshold of that mask
    #       if the pixel is adjescent to multiple masks, only consider the mask to which it is closest in stat norm
    #       if it is, add the pixel to the mask and add the pixel statistics to the mask statistics
    #       if it is not close enough to any mask, we increase the impatience for that pixel 
    #           higher impatience relaxes the distance requirements for the pixel to join a region
    floodfill.fill_dust(mapping, 
                            seg, 
                            masks, 
                            stats, 
                            np.ones(shape=[seg.shape[2]], dtype='float32') * promote_pixel_threshold, 
                            join_threshold, 
                            impatience)         # 0.001 - 0.2 sec
    
    # use statistics to assign each mask a single label, then combine labels together to make a pixel segmentation
    output = poolSegmentation(masks, seg, stats)
    # return the result
    if return_partial:
        return output, partial
    else:
        return output

def main(boundary_threshold = 0.7, boundary_path = './visualization_0_7/', seg_path='./seg3/'):
    # load files from cache
    # for now we are just trying to post process the boundary info
    raw = np.load('./cache/raw.npy').transpose([0,2,3,1])   # 256,512,512,3     uint8
    segmentation = np.load('./cache/segmentation.npy').astype(np.float32)    # 256,24,512,512    float16
    boundary = np.load('./cache/boundary.npy')              # 256, 510, 510     float32
    targets = np.load('./cache/targets.npy')                # 256, 512, 512     uint8

    truth = getBoundaryMask(targets).astype(np.uint8) 

    for i in range(truth.shape[0]):
        print(i)
        t = truth[i,:, :]
        f = boundary[i, :, :] 
        seg = segmentation[i, :, 1:-1,1:-1]
        target = targets[i, 1:-1, 1:-1] 
        visualizeBoundary(f,t,os.path.join(boundary_path, 'img_' + str(i)+ '.png'),boundary_threshold= boundary_threshold)
        processed, partial = postProcess(seg, f, boundary_threshold=boundary_threshold, expand_pixel_size=5, promote_pixel_threshold=0.5, join_threshold=0.05, impatience=1.5, return_partial=True)
        # compute raw
        raw_seg = np.argmax(seg,axis=0).astype(np.uint8)
        logImageSet(os.path.join(seg_path, 'img_' + str(i)+'.png'),raw[i,1:-1,1:-1,:],target, raw_seg, partial, processed)
         

def apply_mask_color(mask, mask_color):
    return np.concatenate(([mask[ ... , np.newaxis] * color for color in mask_color]), axis=2)

if __name__ == '__main__':
    main()