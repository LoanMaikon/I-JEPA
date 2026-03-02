import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# Using as base https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py

"""
Transform an image into a tensor of shape [B, N, D] with all the patches embedded (e.g. tokens)
[B, N, D] = Batch, Tokens, Embedding
"""
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embedding_dimension=768):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dimension = embedding_dimension

        self.num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)

        self.proj = nn.Conv2d(self.in_channels, self.embedding_dimension, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x.shape = [B, C, H, W]
        # self.proj(x).shape = [B, D, H/patch_size, W/patch_size]
        # self.proj(x).flatten(2).shape = [B, D, N]
        # self.proj(x).flatten(2).transpose(1, 2).shape = [B, N, D] = Batch, Tokens, Embedding
        return self.proj(x).flatten(2).transpose(1, 2)

def create_positional_embedding(num_patches, embedding_dimension):
    positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embedding_dimension), requires_grad=False)
    values = get_2d_sincos_pos_embed(embedding_dimension, int(num_patches**0.5), cls_token=False)

    return positional_embedding.data.copy_(torch.from_numpy(values).float().unsqueeze(0))

"""
grid_size: int of the grid height and width
return [grid_size * grid_size, embed_dim] or [1 + grid_size * grid_size, embed_dim] if cls_token is True
"""
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    
    """
    def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   # (D/2,)

        pos = pos.reshape(-1)   # (M,)
        out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

"""
Main class
"""
class TinyViT(nn.Module):
    def __init__(self,
                image_size=[224],
                patch_size=16,
                in_channels=3,
                embedding_dimesion=192,
                predictor_embedding_dimension=384,
                depth=12,
                predictor_depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attention_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=nn.LayerNorm,
                init_std=0.02,
                ):
        super(TinyViT, self).__init__()

        # Divide the image into patches and embed them
        self.tokenizer = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embedding_dimension=embedding_dimesion)

        # Positional embedding (not learnable)
        self.positional_embedding = create_positional_embedding(num_patches=self.tokenizer.num_patches, embedding_dimension=embedding_dimesion)

    def forward(self, x):
        pass
