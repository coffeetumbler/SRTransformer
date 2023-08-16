import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

import models.transformer as transformer_modules

from utils.functions import clone_layer, masking_matrix



# Multi-scale encoder for image restoration
class MultiScaleIREncoder(nn.Module):
    """
    Args:
        img_res : input image resolution for training
        d_embed : embedding dimension for first block
        n_layer : number of layers in each block
        n_head : number of multi-heads for first block
        hidden_dim_rate : rate of dimension of hidden layer in FFNN
        window_size : window size
        dropout : dropout ratio
        path_dropout : dropout ratio for path drop
        sr_upscale : upscale factor for sr task
        version : model version
    """
    def __init__(self, img_res=64, d_embed=152, n_layer=[3,3,3,3,3,3], n_head=4, hidden_dim_rate=3,
                 window_size=8, dropout=0, path_dropout=0, sr_upscale=1, test_version=False):
        super(MultiScaleIREncoder, self).__init__()
        self.d_embed = d_embed
        self.test_version = test_version
        self.sr_upscale = sr_upscale

        self.n_layer = n_layer
        self.n_block = len(n_layer)
        n_layer = sum(n_layer)
        self.n_head = n_head

        self.window_size = window_size
        self.img_res_unit = 2 * window_size
        self.hidden_dim_rate = hidden_dim_rate

        # initial feature mapping layers
        self.initial_feature_mapping = nn.Conv2d(3, d_embed, kernel_size=3, stride=1, padding=1)
        trunc_normal_(self.initial_feature_mapping.weight, std=.02)
        nn.init.zeros_(self.initial_feature_mapping.bias)

        # transformer encoder and decoders
        n_class = 4 if n_head >= 4 else 2
        self_attn_layer = transformer_modules.MultiHeadSelfAttentionLayer(d_embed, n_head, img_res, img_res, window_size, n_class)
        cross_attn_layer = transformer_modules.ConvolutionalValuedAttentionLayer(d_embed, 2*d_embed, n_head*2, img_res, img_res,
                                                                                    window_size*2, img_res//2, img_res//2, window_size)
        ff_layer = transformer_modules.ConvolutionalFeedForwardLayer(d_embed, d_embed*hidden_dim_rate, dropout)
        norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
        key_norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
        decoder_layer = transformer_modules.DecoderLayer(self_attn_layer, cross_attn_layer, ff_layer, norm_layer, key_norm_layer,
                                                            dropout, path_dropout)
        self.block_1 = clone_layer(decoder_layer, n_layer)
        del(decoder_layer)
        
        self.masks_1 = (masking_matrix(n_head, img_res, img_res, window_size, window_size//2, n_class=n_class),
                        masking_matrix(n_head*2, img_res, img_res, window_size*2, window_size,
                                    img_res//2, img_res//2, window_size, window_size//2))

        # feature addition modules
        layers = [nn.LayerNorm(d_embed, eps=1e-6), nn.Linear(d_embed, d_embed)]
        nn.init.ones_(layers[0].weight)
        nn.init.zeros_(layers[0].bias)
        trunc_normal_(layers[1].weight, std=.02)
        nn.init.zeros_(layers[1].bias)
        block_norm_layer = nn.Sequential(*layers)
        self.block_norm_layers = clone_layer(block_norm_layer, self.n_block)
        del(block_norm_layer)

        self.activation_layer = nn.GELU()
        layers = [nn.Conv2d(d_embed, d_embed*hidden_dim_rate, kernel_size=1, stride=1, padding=0),
                  self.activation_layer,
                  nn.Conv2d(d_embed*hidden_dim_rate, d_embed*hidden_dim_rate, kernel_size=3, stride=1,
                            padding=1, groups=d_embed*hidden_dim_rate),
                  self.activation_layer,
                  nn.Conv2d(d_embed*hidden_dim_rate, d_embed, kernel_size=1, stride=1, padding=0)]
        for layer in layers[::2]:
            trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)
        feature_addition_module = nn.Sequential(*layers)
        self.feature_addition_modules = clone_layer(feature_addition_module, self.n_block)
        del(feature_addition_module)

        final_d_embed = d_embed * 2
        
        # reconstruction module by upscaling factor
        if sr_upscale == 1:
            self.reconstruction_module = nn.ModuleList([nn.Conv2d(final_d_embed, 3, kernel_size=3, stride=1, padding=1)])
        elif sr_upscale == 2:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2]
        elif sr_upscale == 3:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 27, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex3]
        elif sr_upscale == 4:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 2*final_d_embed, kernel_size=1, stride=1, padding=0)
            self.activation_layer = nn.GELU()
            self.conv_layer_2 = nn.Conv2d(final_d_embed//2, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2, self.activation_layer,
                                          self.conv_layer_2, self.pixelshufflex2]
            
        for layer in self.reconstruction_module[::3]:
            trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)


    def pixelshufflex2(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 2*H, 2*W)
    
    def pixelshufflex3(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 3, 3, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 3*H, 3*W)
    

    def make_mask(self, n_head, H, W, cross_mask=True, device=torch.device('cpu')):
        self_mask = masking_matrix(n_head, H, W, self.window_size, self.window_size//2,
                                   n_class=4 if n_head >= 4 else 2)
        if cross_mask:
            cross_mask = masking_matrix(n_head*2, H, W, self.window_size*2, self.window_size,
                                        H//2, W//2, self.window_size, self.window_size//2)
            return (self_mask.to(device), cross_mask.to(device))
        else:
            return self_mask.to(device)


    def forward(self, lq_img, load_mask=True, **kwargs):
        """
        <input>
            lq_img : (n_batch, 3, lq_img_height, lq_img_width), low-quality image
            
        <output>
            hq_img : (n_batch, 3, hq_img_height, hq_img_width), high-quality image
        """
        _, _, H, W = lq_img.shape
        device = lq_img.device

        # Make an initial feature maps.
        x_init = self.initial_feature_mapping(lq_img).permute(0, 2, 3, 1)  # (B, H, W, C)

        # Load masks.
        if load_mask:
            masks_1 = (self.masks_1[0].to(device), self.masks_1[1].to(device))
        else:
            masks_1 = self.make_mask(self.n_head, H, W, True, device)

        # Encode multi-scale features.
        x_1 = x_init
        if not self.test_version:
            n_layer = 0
            for i, block in enumerate(self.n_layer):
                x = self.block_norm_layers[i](x_1)
                for _ in range(block):
                    attention_layer = self.block_1[n_layer]
                    x = attention_layer.self_attention(x, masks_1[0])
                    x = attention_layer.cross_attention(x, x, masks_1[1], None)
                    x = attention_layer.feed_forward(x)
                    n_layer += 1
                x_1 = x_1 + self.feature_addition_modules[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            n_layer = 0
            for i, block in enumerate(self.n_layer):
                x = self.block_norm_layers[i](x_1)
                for _ in range(block):
                    x = self.block_1[n_layer](x, x, *masks_1)
                    n_layer += 1
                x_1 = x_1 + self.feature_addition_modules[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = torch.cat((x_init, x_1), dim=-1).permute(0, 3, 1, 2)

        # Reconstruct image.
        for layer in self.reconstruction_module:
            x = layer(x)
        return x
    

    def evaluate(self, lq_img):
        _, _, H, W = lq_img.shape

        # Set padding options.
        H_pad = ((H - 1) // self.img_res_unit + 1) * self.img_res_unit
        W_pad = ((W - 1) // self.img_res_unit + 1) * self.img_res_unit

        h_pad = H_pad - H
        w_pad = W_pad - W

        # Pad boundaries.
        pad_img = F.pad(lq_img, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Restore image.
        hq_img = self.forward(pad_img, False)
        return hq_img[..., :H*self.sr_upscale, :W*self.sr_upscale]
    

    def flops(self, height, width):
        flops = 0
        # number of pixels
        N = height * width

        # initial feature mapping
        flops += N * self.d_embed * 3 * 9

        # all attention layers
        n_layer = sum(self.n_layer)
        flops += n_layer * self.block_1[0].flops(N, N)

        # sub-layers for residual connection
        hidden_dim = self.d_embed * self.hidden_dim_rate
        norm_linear_flops = N * self.d_embed + N * (self.d_embed ** 2)
        feature_addition_flops = 2 * N * self.d_embed * hidden_dim + 9 * N * hidden_dim
        flops += self.n_block * (norm_linear_flops + feature_addition_flops)
        
        # reconstruction
        final_d_embed = 2 * self.d_embed
        if self.sr_upscale == 1:
            flops += N * final_d_embed * 3 * 9
        elif self.sr_upscale == 2:
            flops += N * final_d_embed * 12 * 9
        elif self.sr_upscale == 3:
            flops += N * final_d_embed * 27 * 9
        elif self.sr_upscale == 4:
            flops += N * (final_d_embed ** 2) * 2
            flops += N * final_d_embed//2 * 12 * 9

        return flops
    



# IREncoder with Dilated Windowed Self-Attention
class DilatedWindowedIREncoder(nn.Module):
    """
    Args:
        img_res : input image resolution for training
        d_embed : embedding dimension for first block
        n_layer : number of layers in each block
        n_head : number of multi-heads for first block
        hidden_dim_rate : rate of dimension of hidden layer in FFNN
        window_size : window size
        dropout : dropout ratio
        path_dropout : dropout ratio for path drop
        sr_upscale : upscale factor for sr task
        version : model version
    """
    def __init__(self, img_res=64, d_embed=152, n_layer=[3,3,3,3,3,3], n_head=4, hidden_dim_rate=3,
                 conv_hidden_rate=2, window_size=8, dropout=0, path_dropout=0, sr_upscale=1):
        super(DilatedWindowedIREncoder, self).__init__()
        self.d_embed = d_embed
        self.sr_upscale = sr_upscale

        self.n_layer = sum(n_layer)
        self.n_block = len(n_layer)
        self.n_head = n_head

        self.window_size = window_size
        self.hidden_dim_rate = hidden_dim_rate

        # initial feature mapping layers
        self.initial_feature_mapping = nn.Conv2d(3, d_embed, kernel_size=3, stride=1, padding=1)
        trunc_normal_(self.initial_feature_mapping.weight, std=.02)
        nn.init.zeros_(self.initial_feature_mapping.bias)

        # transformer encoders
        for i, _n_layer in enumerate(n_layer):
            block = transformer_modules.get_transformer_encoder(d_embed=d_embed,
                                                                attention_type='dilated',
                                                                convolutional_ff=True,
                                                                n_layer=_n_layer,
                                                                n_head=n_head,
                                                                d_ff=d_embed*hidden_dim_rate,
                                                                conv_hidden_rate=conv_hidden_rate,
                                                                n_patch=img_res,
                                                                window_size=window_size,
                                                                dropout=dropout,
                                                                path_dropout=path_dropout)
            setattr(self, 'block_{}'.format(i+1), block)

        # feature addition modules
        layers = [nn.LayerNorm(d_embed, eps=1e-6), nn.Linear(d_embed, d_embed)]
        nn.init.ones_(layers[0].weight)
        nn.init.zeros_(layers[0].bias)
        trunc_normal_(layers[1].weight, std=.02)
        nn.init.zeros_(layers[1].bias)
        block_norm_layer = nn.Sequential(*layers)
        self.block_norm_layers = clone_layer(block_norm_layer, self.n_block)
        del(block_norm_layer)

        self.activation_layer = nn.GELU()
        layers = [nn.Conv2d(d_embed, d_embed*4, kernel_size=1, stride=1, padding=0),
                  self.activation_layer,
                  nn.Conv2d(d_embed*4, d_embed*4, kernel_size=3, stride=1, padding=1, groups=d_embed*4),
                  self.activation_layer,
                  nn.Conv2d(d_embed*4, d_embed, kernel_size=1, stride=1, padding=0)]
        for layer in layers[::2]:
            trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)
        feature_addition_module = nn.Sequential(*layers)
        self.feature_addition_modules = clone_layer(feature_addition_module, self.n_block)
        del(feature_addition_module)

        final_d_embed = d_embed * 2
        
        # reconstruction module by upscaling factor
        if sr_upscale == 1:
            self.reconstruction_module = nn.ModuleList([nn.Conv2d(final_d_embed, 3, kernel_size=3, stride=1, padding=1)])
        elif sr_upscale == 2:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2]
        elif sr_upscale == 3:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 27, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex3]
        elif sr_upscale == 4:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 2*final_d_embed, kernel_size=1, stride=1, padding=0)
            self.activation_layer = nn.GELU()
            self.conv_layer_2 = nn.Conv2d(final_d_embed//2, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2, self.activation_layer,
                                          self.conv_layer_2, self.pixelshufflex2]
            
        for layer in self.reconstruction_module[::3]:
            trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)


    def pixelshufflex2(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 2*H, 2*W)
    
    def pixelshufflex3(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 3, 3, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 3*H, 3*W)


    def forward(self, lq_img):
        """
        <input>
            lq_img : (n_batch, 3, lq_img_height, lq_img_width), low-quality image
            
        <output>
            hq_img : (n_batch, 3, hq_img_height, hq_img_width), high-quality image
        """
        # Make an initial feature maps.
        x_init = self.initial_feature_mapping(lq_img).permute(0, 2, 3, 1)  # (B, H, W, C)

        # Encode features.
        x_1 = x_init
        for i in range(self.n_block):
            x = self.block_norm_layers[i](x_1)
            x = getattr(self, 'block_{}'.format(i+1))(x)
            x_1 = x_1 + self.feature_addition_modules[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.cat((x_init, x_1), dim=-1).permute(0, 3, 1, 2)

        # Reconstruct image.
        for layer in self.reconstruction_module:
            x = layer(x)
        return x
    

    def evaluate(self, lq_img):
        _, _, H, W = lq_img.shape

        # Set padding options.
        H_pad = ((H - 1) // self.window_size + 1) * self.window_size
        W_pad = ((W - 1) // self.window_size + 1) * self.window_size

        h_pad = H_pad - H
        w_pad = W_pad - W

        # Pad boundaries.
        pad_img = F.pad(lq_img, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Restore image.
        hq_img = self.forward(pad_img)
        return hq_img[..., :H*self.sr_upscale, :W*self.sr_upscale]
    

    def flops(self, height, width):
        flops = 0
        # number of pixels
        N = height * width

        # initial feature mapping
        flops += N * self.d_embed * 3 * 9

        # all encoder blocks
        for i in range(1, self.n_block+1):
            flops += getattr(self, 'block_{}'.format(i)).flops(N)

        # sub-layers for residual connection
        hidden_dim = self.d_embed * 4
        norm_linear_flops = N * self.d_embed + N * (self.d_embed ** 2)
        feature_addition_flops = 2 * N * self.d_embed * hidden_dim + 9 * N * hidden_dim
        flops += self.n_block * (norm_linear_flops + feature_addition_flops)
        
        # reconstruction
        final_d_embed = 2 * self.d_embed
        if self.sr_upscale == 1:
            flops += N * final_d_embed * 3 * 9
        elif self.sr_upscale == 2:
            flops += N * final_d_embed * 12 * 9
        elif self.sr_upscale == 3:
            flops += N * final_d_embed * 27 * 9
        elif self.sr_upscale == 4:
            flops += N * (final_d_embed ** 2) * 2
            flops += N * final_d_embed//2 * 12 * 9

        return flops
    



# IREncoder with Expanded Windowed Self-Attention
class ExpandedWindowedIREncoder(nn.Module):
    """
    Args:
        img_res : input image resolution for training
        d_embed : embedding dimension for first block
        n_layer : number of layers in each block
        n_head : number of multi-heads for first block
        hidden_dim_rate : rate of dimension of hidden layer in FFNN
        window_size : window size
        dropout : dropout ratio
        path_dropout : dropout ratio for path drop
        sr_upscale : upscale factor for sr task
        version : model version
    """
    def __init__(self, img_res=64, d_embed=152, n_layer=[3,3,3,3,3,3], n_head=4, hidden_dim_rate=3,
                 conv_hidden_rate=2, residual_hidden_rate=4, window_size=8, dropout=0, path_dropout=0, sr_upscale=1):
        super(ExpandedWindowedIREncoder, self).__init__()
        if isinstance(d_embed, int):
            self.d_embed = d_embed
            self.d_embed_list = [d_embed] * len(n_layer)
        else:
            self.d_embed = max(d_embed)
            self.d_embed_list = d_embed
        self.sr_upscale = sr_upscale

        self.n_layer = sum(n_layer)
        self.n_block = len(n_layer)
        self.n_head = n_head

        self.window_size = window_size
        self.hidden_dim_rate = hidden_dim_rate
        self.residual_hidden_rate = residual_hidden_rate

        # initial feature mapping layers
        self.initial_feature_mapping = nn.Conv2d(3, self.d_embed, kernel_size=3, stride=1, padding=1)
        trunc_normal_(self.initial_feature_mapping.weight, std=.02)
        nn.init.zeros_(self.initial_feature_mapping.bias)

        # transformer encoders
        for i, _n_layer, d in zip(range(self.n_block), n_layer, self.d_embed_list):
            block = transformer_modules.get_transformer_encoder(d_embed=d,
                                                                attention_type='expanded',
                                                                convolutional_ff=True,
                                                                n_layer=_n_layer,
                                                                n_head=n_head,
                                                                d_ff=d*hidden_dim_rate,
                                                                conv_hidden_rate=conv_hidden_rate,
                                                                n_patch=img_res,
                                                                window_size=window_size,
                                                                dropout=dropout,
                                                                path_dropout=path_dropout)
            setattr(self, 'block_{}'.format(i+1), block)

        # feature addition modules
        block_norm_layers = []
        for d in self.d_embed_list:
            layers = [nn.LayerNorm(self.d_embed, eps=1e-6), nn.Linear(self.d_embed, d)]
            nn.init.ones_(layers[0].weight)
            nn.init.zeros_(layers[0].bias)
            trunc_normal_(layers[1].weight, std=.02)
            nn.init.zeros_(layers[1].bias)
            block_norm_layers.append(nn.Sequential(*layers))
        self.block_norm_layers = nn.ModuleList(block_norm_layers)
        del(block_norm_layers)

        self.activation_layer = nn.GELU()

        feature_addition_modules = []
        for d in self.d_embed_list:
            hidden_d = d * residual_hidden_rate
            layers = [nn.Conv2d(d, hidden_d, kernel_size=1, stride=1, padding=0),
                      self.activation_layer,
                      nn.Conv2d(hidden_d, hidden_d, kernel_size=3, stride=1, padding=1, groups=hidden_d),
                      self.activation_layer,
                      nn.Conv2d(hidden_d, self.d_embed, kernel_size=1, stride=1, padding=0)]
            for layer in layers[::2]:
                trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)
            feature_addition_modules.append(nn.Sequential(*layers))
        self.feature_addition_modules = nn.ModuleList(feature_addition_modules)
        del(feature_addition_modules)

        final_d_embed = self.d_embed * 2
        
        # reconstruction module by upscaling factor
        if sr_upscale == 1:
            self.reconstruction_module = nn.ModuleList([nn.Conv2d(final_d_embed, 3, kernel_size=3, stride=1, padding=1)])
        elif sr_upscale == 2:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2]
        elif sr_upscale == 3:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 27, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex3]
        elif sr_upscale == 4:
            self.conv_layer_1 = nn.Conv2d(final_d_embed, 2*final_d_embed, kernel_size=1, stride=1, padding=0)
            self.activation_layer = nn.GELU()
            self.conv_layer_2 = nn.Conv2d(final_d_embed//2, 12, kernel_size=3, stride=1, padding=1)
            self.reconstruction_module = [self.conv_layer_1, self.pixelshufflex2, self.activation_layer,
                                          self.conv_layer_2, self.pixelshufflex2]
            
        for layer in self.reconstruction_module[::3]:
            trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)


    def pixelshufflex2(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 2*H, 2*W)
    
    def pixelshufflex3(self, img):
        B, _, H, W = img.shape
        return img.reshape(B, -1, 3, 3, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, -1, 3*H, 3*W)


    def forward(self, lq_img, load_mask=True):
        """
        <input>
            lq_img : (n_batch, 3, lq_img_height, lq_img_width), low-quality image
            
        <output>
            hq_img : (n_batch, 3, hq_img_height, hq_img_width), high-quality image
        """
        # Make an initial feature maps.
        x_init = self.initial_feature_mapping(lq_img).permute(0, 2, 3, 1)  # (B, H, W, C)

        # Encode features.
        x_1 = x_init
        if load_mask:
            for i in range(self.n_block):
                x = self.block_norm_layers[i](x_1)
                x = getattr(self, 'block_{}'.format(i+1))(x)
                x_1 = x_1 + self.feature_addition_modules[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            for i in range(self.n_block):
                x = self.block_norm_layers[i](x_1)
                x = getattr(self, 'block_{}'.format(i+1))(x, False)
                x_1 = x_1 + self.feature_addition_modules[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.cat((x_init, x_1), dim=-1).permute(0, 3, 1, 2)

        # Reconstruct image.
        for layer in self.reconstruction_module:
            x = layer(x)
        return x
    

    def evaluate(self, lq_img):
        _, _, H, W = lq_img.shape

        # Set padding options.
        H_pad = ((H - 1) // self.window_size + 1) * self.window_size
        W_pad = ((W - 1) // self.window_size + 1) * self.window_size

        h_pad = H_pad - H
        w_pad = W_pad - W

        # Pad boundaries.
        pad_img = F.pad(lq_img, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Restore image.
        hq_img = self.forward(pad_img, False)
        return hq_img[..., :H*self.sr_upscale, :W*self.sr_upscale]
    

    def flops(self, height, width):
        flops = 0
        # number of pixels
        N = height * width

        # initial feature mapping
        flops += N * self.d_embed * 3 * 9

        # all encoder blocks
        for i in range(1, self.n_block+1):
            flops += getattr(self, 'block_{}'.format(i)).flops(N)

        # sub-layers for residual connection
        for d in self.d_embed_list:
            hidden_dim = d * self.residual_hidden_rate
            norm_linear_flops = N * self.d_embed + N * self.d_embed * d
            feature_addition_flops = N * (self.d_embed + d) * hidden_dim + 9 * N * hidden_dim
            flops += norm_linear_flops + feature_addition_flops
        
        # reconstruction
        final_d_embed = 2 * self.d_embed
        if self.sr_upscale == 1:
            flops += N * final_d_embed * 3 * 9
        elif self.sr_upscale == 2:
            flops += N * final_d_embed * 12 * 9
        elif self.sr_upscale == 3:
            flops += N * final_d_embed * 27 * 9
        elif self.sr_upscale == 4:
            flops += N * (final_d_embed ** 2) * 2
            flops += N * final_d_embed//2 * 12 * 9

        return flops