import math
import torch
import torch.nn as nn
import numpy as np

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

class PositionEncodingBlock2D(nn.Module):
    def __init__(self, d_input: int, height: int, width: int):
        super().__init__()
        with torch.no_grad():
            positions1D = getPositionEncoding(max(height, width),4) 
            horizontal = torch.broadcast_to(torch.unsqueeze(torch.tensor(positions1D[0: width, :]), 1), (width, height, 4))
            vertical =  torch.broadcast_to(torch.unsqueeze(torch.tensor(positions1D[0: height, :]), 0), (width, height, 4))
            # (W, H, 4) + (W, H, 4) -> (W, H, 8) -> (W, H, d_input) -> (d_input, H, W) -> (1, d_input, H, W) 
            self.register_buffer('positional_embedding', torch.unsqueeze(torch.transpose(nn.functional.pad(torch.cat((horizontal, vertical), 2), (0, d_input - 8)), 0, -1),0)) 
    def forward(self, x):
        return x + self.positional_embedding.to(dtype = x.dtype)

class AttentionBlock2D(nn.Module):
    # we will use fixed position embeddings for this
    def __init__(self, d_input: int, n_heads:int, d_kq:int, d_val: int, d_output:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.kq_proj = nn.Linear(d_input, 2 * n_heads * d_kq, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_input, n_heads * d_val, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_kq = d_kq
        self.d_val = d_val
        self.out_proj = nn.Linear(n_heads * d_val, d_output)
           
    def forward(self, x: torch.Tensor):
        batch, channels, height, width = x.shape
        x = x.transpose(1, -1) # B, W, H, C 
        q, k = torch.chunk(self.kq_proj(x),2,-1) 
        v = self.v_proj(x)
        q = torch.reshape(q, (batch, width*height, self.n_heads, self.d_kq)).transpose(-2, -3) # B, heads, W*H, d_kq
        k = torch.reshape(k, (batch, width*height, self.n_heads, self.d_kq)).transpose(-2, -3) # B, heads, W*H, d_kq
        v = torch.reshape(v, (batch, width*height, self.n_heads, self.d_val)).transpose(-2, -3) # B, heads, W*H, d_val

        values = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=0, is_causal=False, scale=None)
        #weights = q @ k.transpose(-1, -2) # B, heads, W*H, W*H
        #weights /= self.d_kq ** 0.5 # normalize weights by dimension
        #logits = nn.functional.softmax(weights,-1)
        #values = logits @ v # B, heads, W*H, d_val
        values = torch.reshape(values.transpose(1, 2), (batch, width, height, self.n_heads * self.d_val)) # B, W, H, n_heads * d_val
        output = self.out_proj(values) # B, W, H, d_output
        return output.transpose(1, -1) # B, d_output, H, W 
    
    def init_weights(self):
        for module in (self.kq_proj, self.v_proj, self.out_proj):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                module.bias.data.fill_(0.01)

# I'm actually quite curious about the "shape" of the kernel learned by this type of network 
class AttentionConvolution2D(nn.Module):
    def __init__(self, n_heads:int, d_head:int, d_input:int, d_output:int, block_size:int, height:int, width:int, in_proj_bias=True, out_proj_bias=True, learnable_cross_embedding=True, cross_embed_alpha=2.0, cross_embed_beta = 1.0/16, cross_embed_threshold = math.inf):
        super().__init__()
        # stack the query, key, and value projections together, within each one stack each head
        self.in_proj = nn.Linear(d_input, 3 * n_heads * d_head, bias=in_proj_bias)

        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_head * n_heads, d_output, bias=out_proj_bias)

        # keep track of interal dimensions
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_output = d_output

        self.block_size = block_size

        # prepare a positional embedding penalty term for each head
        # cross_embed_alpha: power to use for horizontal and vertical displacement
        # cross_embed_beta: scale parameter to use
        # cross_embed_threshold: distances exceeding this value are initialized to infinity instead.
        #   this creates a local mask for the convolution, the bounds of this mask are not learnable by the network
        #   also it may fuck everything up, I don't know 
        # use these together to create a potential well that defines the kernel of each head 
        # parameters are made learnable by default  

        # horizontal majour, vertical minor dimensions 
        self.positional_cross_embedding = nn.Parameter(torch.zeros((n_heads,block_size * 4 * block_size * 4)), requires_grad=learnable_cross_embedding)
        with torch.no_grad():
            for x in range(block_size * 4):
                for y in range(block_size * 4):
                    value =  cross_embed_beta* (abs(x - 2 * block_size)**cross_embed_alpha + abs(y - 2 * block_size)**cross_embed_alpha)
                    self.positional_cross_embedding[:, 4 * block_size * x + y] = value if value < cross_embed_threshold else math.inf
        
            # compute mask components, this is slow but only happens once
            masks = torch.zeros((block_size * block_size * 16, block_size * block_size, 9 * block_size * block_size), requires_grad=False) 
            for x_0 in range (block_size):
                for y_0 in range(block_size):
                    for x_1 in range (3 * block_size):
                        for y_1 in range(3 * block_size):
                            x_net = x_1 + block_size - x_0
                            y_net = y_1 + block_size - y_0
                            masks[4 * block_size * x_net + y_net, block_size * x_0 + y_0, 3 * block_size * x_1 + y_1] = 1.0 

            self.register_buffer('masks', masks)

            # compute the cutoff mask here so we don't have to move large vectors around in memory
            # this is the only reason we need the height and width in advance
            cutoff = torch.zeros((1, width//block_size, height//block_size, self.n_heads, self.block_size * self.block_size, 3 * self.block_size, 3 * self.block_size))
            cutoff[:, 0,  :, :, :, 0: self.block_size, : ] = torch.inf 
            cutoff[:, -1, :, :, :, 2 * self.block_size : , : ] = torch.inf 
            cutoff[:, :, 0,  :, :, :, 0: self.block_size] = torch.inf 
            cutoff[:, :, -1, :, :, :, 2 * self.block_size : ] = torch.inf 
            cutoff = cutoff.reshape(1, width//block_size, height//block_size, self.n_heads, self.block_size * self.block_size, 3 * self.block_size * 3 * self.block_size)

            self.register_buffer('cutoff', cutoff)

    def init(self):
        for module in (self.in_proj, self.out_proj):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                module.bias.data.fill_(0.01)

    def forward(self, x):
        # store height and width for computations, should be divisible by block_size
        batch_size = x.shape[0]
        d_input = x.shape[1]
        height = x.shape[2]
        vertical_blocks = height // self.block_size

        width = x.shape[3] 
        horizontal_blocks = width // self.block_size
        # x: # (Batch_Size, d_input, height, width)
        x = torch.transpose(x, 1, -1)               # Batch_size, width, height, d_input
        x = torch.reshape(x, (batch_size, horizontal_blocks, self.block_size, vertical_blocks, self.block_size, d_input)) 
        x = torch.transpose(x, 2, 3)    # batch_size, horizontal block, vertical block, horizontal index, vertical index, d_input 
        # x = torch.reshape(x, (batch_size , horizontal_blocks , vertical_blocks, self.block_size * self.block_size, d_input))  #fuse the the index, we don't have to do that yet
        q, kv = torch.tensor_split(self.in_proj(x),torch.tensor([self.n_heads*self.d_head], dtype=torch.long), -1) #3 tensors of dimension  (batch_size, horizontal block, vertical block, horizontal index, vertical index, self.d_head * self.num_heads)
        # pad k,v with an extra BLOCKS in all spatial directions
        kv = torch.nn.functional.pad(kv,(0,0,0,0,0,0,1,1,1,1), value=0)
        # this involves a bunch of data duplication so hopefully cuda can figure something out
        
        # had these backwards this might change a bunch actually. 
        # stack 3 adjescent blocks across horizontal index (now 3 * block_size) 
        kv = torch.cat((kv[:,0:-2,:,:,:],kv[:,1:-1,:,:,:],kv[:,2:,:,:,:]),-3) 
        # now stack them vertically across vertical index (now 3 * block_size) 
        kv = torch.cat((kv[:,:,0:-2,:,:],kv[:,:,1:-1,:,:],kv[:,:,2:,:,:]),-2)  


        # we can now take the dot product attention between each block and each neighbouring block
        q = torch.reshape(q, (batch_size, horizontal_blocks, vertical_blocks, self.block_size * self.block_size, self.d_head * self.n_heads))
        kv = torch.reshape(kv, (batch_size, horizontal_blocks, vertical_blocks, 9 * self.block_size * self.block_size, 2 * self.d_head * self.n_heads))
        k,v = torch.tensor_split(kv, torch.tensor([self.d_head * self.n_heads], dtype=torch.long), -1) 
        # pull the heads out
        q = torch.reshape(q,(batch_size, horizontal_blocks, vertical_blocks, self.block_size * self.block_size, self.n_heads, self.d_head)).transpose(-2, -3)
        k = torch.reshape(k,(batch_size, horizontal_blocks, vertical_blocks, 9 * self.block_size * self.block_size, self.n_heads, self.d_head)).transpose(-2, -3)
        v = torch.reshape(v,(batch_size, horizontal_blocks, vertical_blocks, 9 * self.block_size * self.block_size, self.n_heads, self.d_head)).transpose(-2, -3)

        # weights = q @ k.transpose(-1, -2) # batch_size, horizontal_blocks, vertical_blocks, n_heads, self.block_size * self.block_size, 9 * self.block_size * self.block_size

        # compute the mask
        mask = torch.tensordot(self.positional_cross_embedding, self.masks, dims=([1],[0]))
        mask = torch.unsqueeze(mask,0)
        mask = torch.unsqueeze(mask,0)
        mask = torch.unsqueeze(mask,0)
        # apply mask as penalty, hope it broadcasts
        # weights -= mask
        # apply penalty to weights that see padding
        # weights -= self.cutoff
        
        # now it's just normal attention
        # weights /= math.sqrt(self.d_head) 

        # take soft max along the key/value direction 
        # weight = nn.functional.softmax(weights, dim=-1) 

        # batch multiplication over the last two dimensions

        # output = weight @ v # batch_size, horizontal_blocks, vertical_blocks, self.n_heads, self.block_size * self.block_size, self.d_head

        output = torch.nn.functional.scaled_dot_product_attention(q,k,v, -(mask + self.cutoff), dropout_p=0, is_causal=False, scale=None)

        output = output.transpose(-2, -3) # batch_size, horizontal_blocks, vertical_blocks, self.block_size * self.block_size, self.n_heads,  self.d_head

        output = output.reshape(batch_size, horizontal_blocks, vertical_blocks, self.block_size * self.block_size, self.n_heads * self.d_head)  

        # self.n_heads * self.d_head -> d_output
        output = self.out_proj(output) 
        
        # now we have to unroll everything
        output = output.reshape(batch_size, horizontal_blocks, vertical_blocks, self.block_size, self.block_size, self.d_output)
        output = output.transpose(2,3)
        output = output.reshape(batch_size, horizontal_blocks * self.block_size, vertical_blocks * self.block_size, self.d_output)
        output = output.transpose(-1, 1) # and we are done, swag! 
        return output
    
    def diagnostics(self):
        print('layer: ')
        print(torch.min(self.positional_cross_embedding, 1))
        print(torch.max(self.positional_cross_embedding, 1))