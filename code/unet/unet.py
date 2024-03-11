from __future__ import annotations
import math
import torch
import torch.nn as nn
from .attention import AttentionBlock2D, PositionEncodingBlock2D, AttentionConvolution2D

from torch.utils.checkpoint import checkpoint


class BCELoss_class_weighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, target: torch.Tensor, weights: tuple[float,float]) -> torch.Tensor:
        ctx.save_for_backward(input, target)
        ctx.weights = weights
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return bce
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input,target = ctx.saved_tensors
        weights = ctx.weights
        x = torch.clamp(input,min=1e-7,max=1-1e-7)
        dydx = -target * weights[1] / x + (1 - target) * weights[0] / (1 - x)
        dydtarget = -torch.log(x) * weights[1] + torch.log(1 - x) * weights[0]
        return grad_output * dydx, grad_output * dydtarget, None

# This does not work well at all, keep it around as an example though
class GroupVectorNorm(nn.Module):
    def __init__(self, num_groups: int, num_features: int, eps=1e-4, affine=False):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        assert num_features % num_groups == 0 
        assert eps > 0
        self.eps = eps
        if affine:
            self.weight = torch.nn.Parameter(torch.ones([num_groups]))
            self.bias = torch.nn.Parameter(torch.zeros([num_groups]))
        else:
            self.weight = 1
            self.bias = 0

    def forward(self, x):
        raise NotImplementedError()
        N, C, H, W = x.shape
        assert C == self.num_features
        y = torch.reshape(x, [N, self.num_groups, self.num_features // self.num_groups, H, W])
        y_norm = torch.norm(y,p = 2,dim=[2,3,4],keepdim=True)
        y = y/(y_norm + self.eps) 
        if isinstance(self.weight,torch.nn.Parameter):
            weight = torch.reshape(self.weight.data, [1, self.num_groups, 1, 1, 1])
            bias = torch.reshape(self.bias.data, [1, self.num_groups, 1, 1, 1])
            y = y * weight + bias
        else:
            # for performance reasons we aren't doing null arithmatic operations
            pass
        y = torch.reshape(y, [N,C,H,W])
        return y

class DownBlock(nn.Module):
    def __init__(self, features_in: int, features_out: int, norm = 'GroupVectorNorm'):
        normClass = nn.GroupNorm if norm == 'GroupNorm' else GroupVectorNorm
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(features_in, features_out, kernel_size=3, stride=1, padding='same'),
            nn.SiLU(),
            normClass(32,features_out, affine=False),
            nn.Conv2d(features_out, features_out, kernel_size=3, stride=1, padding='same'),
            nn.SiLU(),
            normClass(32,features_out, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding = 0),           
        ]) 
    
    def init_weights(self):
        for module in self.layers:
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0,0.1)
                module.bias.data.normal_(0,0.1) 

    def forward(self, x):
        residuals = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, (nn.GroupNorm, GroupVectorNorm)):
                residuals.append(x) 
        return x, torch.cat(residuals, dim = 1)
    
class SkipConnectionUp(nn.Module):
    def __init__(self, features_in: int, width: int, height: int):
        super().__init__()
        self.features_in = features_in
        self.width = width
        self.height = height
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x,(batch_size, self.features_in // 2, self.height * 2, self.width))
        x = x.transpose(-1, -2)
        x = torch.reshape(x,(batch_size, self.features_in // 4, self.width * 2, self.height * 2))
        x = x.transpose(-1, -2) 
        x = torch.cat((x,x), 1) 
        return x


class UpBlock(nn.Module):
    def __init__(self, features_in: int, features_out: int, residual_features: int, n_heads: int, d_head: int, block_size: int, height: int, width: int, useAttention = False, norm="GroupVectorNorm"):
        super().__init__()
        normClass = nn.GroupNorm if norm == 'GroupNorm' else GroupVectorNorm
        self.skipUp = nn.UpsamplingNearest2d(scale_factor=2) if features_in <= features_out else SkipConnectionUp(features_in, height//2, width//2)
        self.convTranspose = nn.ConvTranspose2d(features_in, features_out, kernel_size=2, stride=2, padding=0)
        self.norm1 = normClass(32,features_out, affine=False)
        if useAttention:
            self.attention = AttentionConvolution2D(n_heads, d_head, features_out + residual_features, features_out, block_size, height, width)
        else:
            self.attention = None
            self.attention_vars = [n_heads, d_head, features_out + residual_features, features_out, block_size, height, width]
        self.conv1 = nn.Conv2d(features_out + residual_features, features_out , kernel_size=block_size + 1, stride=1, padding='same')
        self.conv2 = nn.Conv2d(features_out, features_out, kernel_size=block_size + 1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(features_out, features_out, kernel_size=block_size + 1, stride=1, padding='same')
        self.norm2 = normClass(32,features_out, affine=False)
        self.norm3 = normClass(32,features_out, affine=False)
        self.norm4 = normClass(32,features_out, affine=False)
        
    def init_weights(self):
        self.convTranspose.weight.data.normal_(0, 0.1)
        self.convTranspose.bias.data.normal_(0, 0.1)
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv1.bias.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        self.conv2.bias.data.normal_(0, 0.1)
        self.conv3.weight.data.normal_(0, 0.1)
        self.conv3.bias.data.normal_(0, 0.1)
        if(self.attention is not None):
            self.attention.init()
            self.attention.out_proj.weight.data *= 0.01
            self.attention.out_proj.bias.data *= 0.01

    def init_attention(self):
        self.attention.init()
    
    def enable_attention(self):
        if self.attention is None:
            self.attention = AttentionConvolution2D(*self.attention_vars)
            self.init_attention()
            self.attention.out_proj.weight.data *= 0.01
            self.attention.out_proj.bias.data *= 0.01

    def forward(self, x, residuals):
        # lossy skip connection over the upsampling
        y = self.skipUp(x) 
        x = self.convTranspose(x)
        if x.shape[1] > y.shape[1]:
            y = nn.functional.pad(y,(0,0,0,0,0,x.shape[1] - y.shape[1]),value=0.0) 
        # put silu after the skip connection so that convTranspose can learn the identity map 
        x = torch.nn.functional.silu(x + y)
        x = self.norm1(x)
        w = self.conv1(torch.cat((x, residuals),1))
        a = self.attention(torch.cat((x, residuals),1)) if self.attention is not None else torch.zeros_like(x)
        x = self.norm2(torch.nn.functional.silu(x + w + a))
        z = self.conv2(x)
        x = self.norm3(torch.nn.functional.silu(x + z))
        z = self.conv3(x)
        x = self.norm4(torch.nn.functional.silu(x + z))
        return x

class ChannelNormalization(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.normalization = nn.LayerNorm(input_features)
    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.normalization(x)
        x = x.transpose(1, -1)
        return x
    def init_weights(self):
        self.normalization.weight.data = torch.ones_like(self.normalization.weight.data) * 3.0

class Unet(nn.Module):
    def __init__(self, input_features: int, output_features: int, height: int, width: int, base_features = 32, bottleneck = "default", useAttention= False, norm="GroupNorm", checkpointing = False):
        super().__init__()
        downBlocks = []
        self.checkpointing = checkpointing
        # keep the same shape for the tail of the model
        downBlocks.append(DownBlock(input_features, base_features *2, norm= norm))
        downBlocks.append(DownBlock(base_features * 2, base_features * 2, norm= norm))
        downBlocks.append(DownBlock(base_features * 2, base_features * 4, norm= norm))
        downBlocks.append(DownBlock(base_features * 4, base_features * 8, norm= norm))
        downBlocks.append(DownBlock(base_features * 8, base_features * 16, norm= norm))
        
        self.downBlocks = nn.ModuleList(downBlocks)

        self.bottleneck = ImprovedBottleNeck(base_features * 16, base_features * 32, 24, 16, 16, norm= norm) if bottleneck == 'improved' else (  
            AttentionBottleNeck(base_features * 16,base_features * 32, 4, 16,16, n_blocks=3, norm= norm) if bottleneck == 'attention' else (
            AttentionBottleNeck(base_features * 16,base_features * 32, 4, 16,16, n_blocks=6, norm= norm) if bottleneck == 'attentionXL' else (
            DefaultBottleNeck(base_features * 16, base_features * 32))
        ))
        upBlocks = []
        upBlocks.append(UpBlock(base_features * 32, base_features * 16, base_features * 32, 4, base_features * 4, 2, height // 16, width // 16, useAttention, norm= norm))
        upBlocks.append(UpBlock(base_features * 16, base_features * 8, base_features * 16, 4, base_features * 2, 2, height // 8, width // 8, useAttention, norm= norm))
        upBlocks.append(UpBlock(base_features * 8, base_features * 4, base_features * 8, 4, base_features * 1, 2, height // 4, width // 4, useAttention, norm= norm))
        upBlocks.append(UpBlock(base_features * 4, base_features * 4, base_features * 4, 4, base_features // 2, 2, height // 2, width // 2, useAttention, norm= norm))
        upBlocks.append(UpBlock(base_features * 4, base_features * 4, base_features * 4, 4, base_features // 2, 2, height, width, useAttention, norm= norm))

        self.upBlocks = nn.ModuleList(upBlocks)
        self.output = nn.Conv2d(base_features * 4, output_features, kernel_size=1, stride=1, padding=0)
        self.normalization = ChannelNormalization(output_features)
        self.boundary = nn.Conv2d(base_features * 4, 1, kernel_size=1, stride=1, padding=0)

    def enable_gradient_checkpointing(self):
        self.checkpointing = True

    def init_weights(self):
        for module in self.downBlocks:
            module.init_weights()
        for module in self.upBlocks:
            module.init_weights()
        
        self.bottleneck.init_weights()
        
        self.output.weight.data.normal_(0, 0.1)
        self.output.bias.data.normal_(0, 0.1)
        self.boundary.weight.data.normal_(0, 0.1)
        self.boundary.bias.data.normal_(0, 0.1)

        self.normalization.init_weights()

    def enable_attention(self):
        module: UpBlock
        for module in self.upBlocks:
            module.enable_attention()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = []
        for module in self.downBlocks:
            if self.checkpointing:
                x, res = checkpoint(module, x, use_reentrant=False)
            else:
                x, res = module(x)
            residuals.append(res)  

        if self.checkpointing:
            x = checkpoint(self.bottleneck, x, use_reentrant=False)
        else:
            x = self.bottleneck(x)
        for module in self.upBlocks:
            res = residuals.pop()
            if self.checkpointing:
                x = checkpoint(module, x, res, use_reentrant=False)
            else:
                x = module(x, res)
        # compute segmentation portion
        seg = self.output(x)
        seg = self.normalization(seg)
        # compute boundary portion
        bdry = self.boundary(x)
        bdry = nn.functional.sigmoid(bdry)
        # return both
        return seg, bdry

class DefaultBottleNeck(nn.Sequential):
    def __init__(self, features_in, features_out):
        super().__init__(*[
            nn.Conv2d(features_in, features_out, kernel_size=1, stride=1, padding='same'),
            nn.SiLU(),
            nn.Conv2d(features_out, features_out, kernel_size=1, stride=1, padding='same'),
            nn.SiLU(),
            nn.GroupNorm(32, features_out),
        ])
    def init_weights(self):
        for module in self:
            if isinstance(module, nn.Conv2d): 
                module.weight.data.normal_(0, 0.1)
                module.bias.data.normal_(0, 0.1)
    

class ImprovedBottleNeck(nn.Module):
    def __init__(self, features_in, features_out, global_features, height, width, norm="GroupVectorNorm"):
        super().__init__()
        normClass = nn.GroupNorm if norm == 'GroupNorm' else GroupVectorNorm
        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=1, padding='same', stride = 1)
        self.conv2 = nn.Conv2d(features_out, features_out - global_features, kernel_size=1, padding='same', stride = 1)
        self.downSample = nn.Conv2d(features_out - global_features, global_features, kernel_size=1, padding='same', stride = 1)
        self.linear1 = nn.Linear( height* width* global_features, height*width * global_features)
        self.linear2 = nn.Linear( height* width* global_features, height*width * global_features)
        self.norm = normClass(32,features_out, affine=False)
        self.conv3 = nn.Conv2d(features_out, features_out, kernel_size=1, padding='same', stride = 1)
        self.global_features = global_features
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.silu(x)
        x = self.conv2(x)
        x = nn.functional.silu(x)
        y = self.downSample(x)
        y = torch.flatten(y, 1, -1)
        y = self.linear1(y)
        y = nn.functional.silu(y)
        y = self.linear2(y)
        y = nn.functional.silu(y)
        y = torch.reshape(y, (x.shape[0], self.global_features, x.shape[2], x.shape[3]))
        x = self.conv3(torch.cat((x,y), 1))
        x = nn.functional.silu(x)
        x = self.norm(x)
        return x
    def init_weights(self):
        for x in [self.conv1, self.conv2, self.conv3, self.downSample, self.linear1, self.linear2]:
            x.weight.data.normal_(0, 0.1)
            x.bias.data.normal_(0, 0.1)

# may as welll go ham on the bottleneck, the additional cost is not super high compared to local attention
class AttentionBottleNeck(nn.Module): 
    def __init__(self, features_in, features_out, n_heads, height, width, n_blocks = 3, norm="GroupVectorNorm"):
        super().__init__()
        normClass = nn.GroupNorm if norm == 'GroupNorm' else GroupVectorNorm
        # perform convolution first
        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=3, padding='same', stride = 1)
        self.conv2 = nn.Conv2d(features_out, features_out, kernel_size=3, padding='same', stride = 1)
        # add positional encoding
        self.positional = PositionEncodingBlock2D(features_out, height, width)
        self.norm = normClass(32,features_out, affine=False)
        self.n_heads = n_heads
        blocks = []
        for i in range(n_blocks):
            blocks.append(AttentionBottleNeckBlock(features_out, features_out, n_heads, features_out // (n_heads * 2), features_out // n_heads, norm))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.silu(x)
        x = self.conv2(x)
        x = nn.functional.silu(x)
        x = self.norm(x)
        x = self.positional(x)
        x = self.blocks(x)
        return x
    def init_weights(self):
        for x in [self.conv1, self.conv2]:
            x.weight.data.normal_(0, 0.1)
            x.bias.data.normal_(0, 0.1)
        for x in self.blocks:
            if isinstance(x, AttentionBottleNeckBlock):
                x.init_weights()
    def grow(self, n_blocks):
        features_out = self.conv2.out_channels
        n_heads = self.n_heads
        norm = 'GroupNorm' if isinstance(self.norm, nn.GroupNorm) else 'GroupVectorNorm'
        # convert current module back into a list
        blocks = []
        for module in self.blocks:
            blocks.append(module)
        for i in range(n_blocks):
            b1 = AttentionBottleNeckBlock(features_out, features_out, n_heads, features_out // (n_heads * 2), features_out // n_heads, norm)
            b1.init_weights()
            b1.attention1.out_proj.weight.data *= 0.1
            b1.attention2.out_proj.weight.data *= 0.1
            blocks.append(b1)
        self.blocks = nn.Sequential(*blocks)


class AttentionBottleNeckBlock(nn.Module):
    def __init__(self, features_in, features_out, n_heads, d_kq, d_val, norm="GroupNorm"):
        super().__init__()
        normClass = nn.GroupNorm if norm == 'GroupNorm' else GroupVectorNorm
        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=1, padding='same', stride = 1)
        self.conv2 = nn.Conv2d(features_out, features_out, kernel_size=1, padding='same', stride = 1)
        self.attention1 = AttentionBlock2D(features_out, n_heads, d_kq, d_val, features_out)
        self.attention2 = AttentionBlock2D(features_out, n_heads, d_kq, d_val, features_out)
        self.norm1 = normClass(32,features_out, affine=False)
        self.norm2 = normClass(32,features_out, affine=False)
        self.skip = nn.Identity() if features_in == features_out else nn.Conv2d(features_in, features_out, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        # skip to the end
        skip = self.skip(x)
        # first perform 1x1 convolutions
        x = self.conv1(x)
        x = nn.functional.silu(x)
        x = self.conv2(x)
        x = nn.functional.silu(x)
        x = self.norm1(x)
        # now perform two rounds of self attention with partial skips
        y = self.attention1(x)
        x = nn.functional.silu(x + y)
        y = self.attention2(x)
        x = nn.functional.silu(x + y)
        x = self.norm2(x)
        # return output plus skip
        return x + skip
    def init_weights(self):
        self.attention1.init_weights()
        self.attention2.init_weights()
        for x in [self.conv1, self.conv2]:
            x.weight.data.normal_(0, 0.1)
            x.bias.data.normal_(0, 0.1)
