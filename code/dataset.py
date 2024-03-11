
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms, io
import numpy as np

# Dataset to handle image to image and image to segmented image training
# Function applies a transformation to both input and output consistently in order to fit into the required bounds
# Input must have 3 channels, segmented output should be one channel 

class ImageSegmentationDataset(Dataset):
    def __init__(self, raw_dir: str, seg_dir: str, width =512, height = 512, transform='FitWidth', seed = 42):
        super().__init__()
        self.width = width
        self.height = height
        self.transform = transform
        self.raw_dir = raw_dir
        self.seg_dir = seg_dir
        print(seed)
        self.rng = np.random.default_rng(seed)
        if seg_dir is not None:
            self.seg = [f for f in os.listdir(seg_dir) if os.path.isfile(os.path.join(seg_dir, f)) and f[-4:] == '.png']
            self.raw = [f.replace('_segm.png', '.jpg') for f in self.seg]
            assert all([os.path.isfile(os.path.join(raw_dir, f)) for f in self.raw]), 'raw files missing'
        else:
            self.raw = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f)) and (f[-4:] == '.jpg' or f[-4:] == '.png')]
            self.seg = None

    def set_seed(self, seed = 42):
        self.rng = np.random.default_rng(seed)
    
    def __len__(self):
        return len(self.raw)
    
    def __getitem__(self, idx):
        with torch.no_grad():
            # read the input image, trim off the alpha channel if there is one
            input_image = io.read_image(os.path.join(self.raw_dir,self.raw[idx]), io.ImageReadMode.RGB)
            # read the segmented image or copy over the input if no segmentation data is available 
            output_image = io.read_image(os.path.join(self.seg_dir,self.seg[idx])) if self.seg is not None else torch.clone(input_image)
            # get the raw dimensions
            raw_channels, raw_height, raw_width = input_image.shape[0:3]
            # consistent rng
            rng = self.rng
            if self.transform == 'FitWidth':
                scale = float(self.width)/ raw_width
                new_height = scale * raw_height
                # input should be resized cleanly 
                input_resizer = transforms.Resize((int(new_height), self.width), transforms.InterpolationMode.BILINEAR, antialias=True)
                # output segmentation should have discrete values
                output_resizer = input_resizer if self.seg is None else transforms.Resize((int(new_height), self.width), transforms.InterpolationMode.NEAREST, antialias=False)
                input_image = input_resizer(input_image)
                output_image = output_resizer(output_image)

                if input_image.shape[1] > self.height:
                    # randomly crop off the difference 
                    difference = input_image.shape[1] - self.height
                    uniform_crop = rng.integers(0, difference, 1).item() 
                    input_image = transforms.functional.crop(input_image, uniform_crop, 0, self.height, self.width) 
                    output_image = transforms.functional.crop(output_image, uniform_crop, 0, self.height, self.width) 
                elif input_image.shape[1] < self.height:
                    # randomly offset and pad the difference
                    difference =  self.height - input_image.shape[1]
                    uniform_shift = rng.integers(0, difference, 1).item() 
                    padded_input = torch.zeros((raw_channels, self.height, self.width),dtype =input_image.dtype)
                    padded_input[:,uniform_shift:uniform_shift + input_image.shape[1],:] = input_image
                    padded_output = torch.zeros((output_image.shape[0], self.height, self.width),dtype =output_image.dtype)
                    padded_output[:,uniform_shift:uniform_shift + output_image.shape[1],:] = output_image
                    input_image, output_image = padded_input, padded_output
                return input_image, output_image
            else:
                raise NotImplementedError
            
    # split a percentage of this dataset into a new dataset, the resultant dataset is disjoint and and their union is the original
    # not super efficient but it only runs once so *shrug*
    def split(self, percentage: float, choices : np.ndarray = None):
        other = ImageSegmentationDataset(self.raw_dir, self.seg_dir, self.width, self.height, self.transform)
        rng = self.rng
        if self.seg_dir is None:
            # choose a percentage of entries 
            if choices is None:
                numbers = rng.choice(len(self.raw), size=int(percentage * len(self.raw)), replace=False)
            else:
                numbers = choices
            chosen = [self.raw[x] for x in numbers]
            remainder = [self.raw[y] for y in range(len(self.raw)) if y not in numbers]
            self.raw = remainder
            other.raw = chosen
            return other
        else:
            if choices is None:
                numbers = rng.choice(len(self.seg), size=int(percentage * len(self.seg)), replace=False)
            else:
                numbers = choices
            chosen_seg = [self.seg[x] for x in numbers]
            chosen_raw = [self.raw[x] for x in numbers]
            remainder_seg = [self.seg[y] for y in range(len(self.seg)) if y not in numbers]
            remainder_raw = [self.raw[y] for y in range(len(self.seg)) if y not in numbers]
            self.raw = remainder_raw
            self.seg = remainder_seg
            other.raw = chosen_raw
            other.seg = chosen_seg
            return other

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
