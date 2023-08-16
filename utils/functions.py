import numpy as np
import copy

import torch
import torch.nn as nn



# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Partition windows of splited heads.
def partition_window(x, window_size):
    """
    <input>
        x : (n_batch, n_head, H, W, C)
        window_size : (int)
        nh : (int) num_window_height
        nw : (int) num_window_width
    
    <return>
        windows : (n_batch, n_head, num_window_height, num_window_width, window_size^2, C)
    """
    n_batch, n_head, H, W, C = x.shape
    nh, nw = H // window_size, W // window_size
    x = x.view(n_batch, n_head, nh, window_size, nw, window_size, C)
    return x.transpose(3, 4).contiguous().view(n_batch, n_head, nh, nw, window_size*window_size, C)


# Merge windows of splited heads.
def merge_window(windows, window_size):
    """
    <input>
        windows : (n_batch, n_head, num_window_height, num_window_width, window_size^2, C)
        window_size : (int)
    
    <return>
        x : (n_batch, n_head, H, W, C)
    """
    n_batch, n_head, nh, nw, N, C = windows.shape
    windows = windows.view(n_batch, n_head, nh, nw, window_size, window_size, C)
    windows = windows.transpose(3, 4).contiguous()
    return windows.view(n_batch, n_head, nh*window_size, nw*window_size, C)


# 4-class cyclic shifting
def cyclic_shift(x, shift_size, n_class=4):
    """
    <input>
        x : (n_batch, n_head, H, W, C)
        shift_size : (int)
        n_class : number of classes for window shifting
    """
    if n_class == 4:
        n_batch, n_head, H, W, C = x.shape
        x = x.view(n_batch, 4, n_head // 4, H, W, C).transpose(0, 1).contiguous()
        x_1 = torch.roll(x[1], shifts=shift_size, dims=-2)
        x_2 = torch.roll(x[2], shifts=shift_size, dims=-3)
        x_3 = torch.roll(x[3], shifts=(shift_size, shift_size), dims=(-2, -3))
        return torch.stack([x[0], x_1, x_2, x_3]).transpose(0, 1).contiguous().view(n_batch, n_head, H, W, C)
    elif n_class == 2:
        n_batch, n_head, H, W, C = x.shape
        x = x.view(n_batch, 2, n_head // 2, H, W, C).transpose(0, 1).contiguous()
        x_1 = torch.roll(x[1], shifts=(shift_size, shift_size), dims=(-2, -3))
        return torch.stack([x[0], x_1]).transpose(0, 1).contiguous().view(n_batch, n_head, H, W, C)
    elif n_class == -2:
        n_batch, n_head, H, W, C = x.shape
        x = x.view(n_batch, 2, n_head // 2, H, W, C).transpose(0, 1).contiguous()
        x_0 = torch.roll(x[1], shifts=shift_size, dims=-2)
        x_1 = torch.roll(x[2], shifts=shift_size, dims=-3)
        return torch.stack([x_0, x_1]).transpose(0, 1).contiguous().view(n_batch, n_head, H, W, C)
    elif n_class == 1:
        return torch.roll(x, shifts=(shift_size, shift_size), dims=(-2, -3))


# Make masking matrix for 4-class split heads.
def masking_matrix(n_head, H, W, window_size, shift_size,
                   H_2=None, W_2=None, window_size_2=None, shift_size_2=None,
                   n_class=4):
    """
    <input>
        n_head, H, W, window_size, shift_size : (int)
        (optional) H_2, W_2, window_size_2, shift_size_2 : (int) key configs, when key != query
        n_class : number of classes for window shifting
        
    <return>
        masking_heads : (1, n_head, num_window_height, num_window_width, window_size^2, window_size^2)
                        or (1, n_head, num_window_height, num_window_width, window_size^2, window_size_2^2)
    """
    # Check if numbers of windows are the same when key != query.
    if H_2 != None:
        assert H_2 // window_size_2 == H // window_size
        assert W_2 // window_size_2 == W // window_size
    
    # Partitioned regions for query.
    if n_class == 4:
        masking_heads_query = torch.zeros(4, H, W, dtype=int)
        masking_heads_query[[1,3], :, :shift_size] = 1
        masking_heads_query[[2,3], :shift_size] += 2
    elif n_class == 2:
        masking_heads_query = torch.zeros(2, H, W, dtype=int)
        masking_heads_query[1, :, :shift_size] = 1
        masking_heads_query[1, :shift_size] += 2
    elif n_class == -2:
        masking_heads_query = torch.zeros(2, H, W, dtype=int)
        masking_heads_query[0, :, :shift_size] = 1
        masking_heads_query[1, :shift_size] = 1
    elif n_class == 1:
        masking_heads_query = torch.zeros(1, H, W, dtype=int)
        masking_heads_query[0, :, :shift_size] = 1
        masking_heads_query[0, :shift_size] += 2

    masking_heads_query = partition_window(masking_heads_query.unsqueeze(0).unsqueeze(-1), window_size)
    
    # Partitioned regions for key.
    if H_2 == None:
        masking_heads_key = masking_heads_query
    else:
        if n_class == 4:
            masking_heads_key = torch.zeros(4, H_2, W_2, dtype=int)
            masking_heads_key[[1,3], :, :shift_size_2] = 1
            masking_heads_key[[2,3], :shift_size_2] += 2
        elif n_class == 2:
            masking_heads_key = torch.zeros(2, H_2, W_2, dtype=int)
            masking_heads_key[1, :, :shift_size_2] = 1
            masking_heads_key[1, :shift_size_2] += 2
        elif n_class == -2:
            masking_heads_key = torch.zeros(2, H_2, W_2, dtype=int)
            masking_heads_key[0, :, :shift_size_2] = 1
            masking_heads_key[1, :shift_size_2] = 1
        elif n_class == 1:
            masking_heads_key = torch.zeros(1, H_2, W_2, dtype=int)
            masking_heads_key[0, :, :shift_size_2] = 1
            masking_heads_key[0, :shift_size_2] += 2

        masking_heads_key = partition_window(masking_heads_key.unsqueeze(0).unsqueeze(-1), window_size_2)
    
    # Create valid masks for heads.
    masking_heads = masking_heads_query - masking_heads_key.transpose(-1, -2)
    masking_heads = masking_heads != 0
    
    return masking_heads.repeat_interleave(n_head // n_class, dim=1)


# Indices for 2D relative position bias for windows
def relative_position_index(window_size, key_window_size=None, qk_ratio=None):
    """
    <input>
        window_size : (int)
        (optional) key_window_size : (int) when key != query
        
    <return>
        relative_coord : (window_size^4, ) or (window_size^2 * key_window_size^2, )
    """
    # Ratio of query to key
    if key_window_size == None:
        qk_ratio = 1
        axis_size = 2 * window_size - 1  # Number of possible positions along an axis
    else:
        if qk_ratio == None:
            assert window_size % key_window_size == 0
            qk_ratio = window_size // key_window_size
        # Number of possible positions along an axis
        axis_size = (window_size + qk_ratio * key_window_size) - qk_ratio
    
    # Coordinate indices along each axis
    query_coord_x = np.repeat(np.arange(window_size) * axis_size, window_size)
    query_coord_y = np.tile(np.arange(window_size), window_size)
    if key_window_size == None:
        key_coord_x = query_coord_x
        key_coord_y = query_coord_y
    else:
        key_coord_x = np.repeat(np.arange(key_window_size) * axis_size * qk_ratio, key_window_size)
        key_coord_y = np.tile(np.arange(key_window_size) * qk_ratio, key_window_size)
        
    # Relative coordinate indices along each axis
    relative_x = query_coord_x[:, np.newaxis] - key_coord_x
    relative_y = query_coord_y[:, np.newaxis] - key_coord_y
    
    # Relative coordinate indices in 2D window
    relative_coord = relative_x + relative_y
    relative_coord -= relative_coord[0, -1]
    
    return relative_coord.flatten()


# Simple upscaling function
def simple_upscale(img, upscale):
    """
    <input>
        img : (..., H, W)
        upscale : (int) upscale factor
        
    <return>
        img : (..., upscale*H, upscale*W)
    """
    return img.repeat_interleave(upscale, dim=-1).repeat_interleave(upscale, dim=-2)