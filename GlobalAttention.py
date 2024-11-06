import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from typing import Optional

# device = "cpu"

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# class Transformer(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_layers):
#         super(Transformer, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
#             num_layers=num_layers
#         )
#         self.fc = nn.Linear(d_model, vocab_size)
#
#     def forward(self, src):
#         src = self.embedding(src)
#         output = self.transformer_encoder(src)
#         output = self.fc(output[-1, :, :])  # Take the output from the last layer for classification
#         return output



class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, num_layers):
        super(VisionTransformer, self).__init__()

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(16, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))  # Remove the +1 here

        # Transformer layers
        transformer_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)

        # Classification head (not included in this example)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)

        # Flatten the spatial dimensions
        x = x.flatten(2).transpose(1, 2)

        # Add positional encoding
        x = x + self.positional_encoding  # This should now match in dimensions

        # Transformer layers
        x = self.transformer(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def func_attention(query, context, gamma1):
    query = query.unsqueeze(0)
    context = context.unsqueeze(0)
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    # print(contextT.shape, query.shape)torch.Size([1, 50176, 3]) torch.Size([1, 11, 77])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contextT = contextT.to(device)
    query = query.to(device)
    # query = query.unsqueeze(0)
    # linear_layer = nn.Linear(11, 3, dtype=torch.float32).to(device)
    query = query.to(torch.float32)
    # query = linear_layer(query)
    # query = query.unsqueeze(0)
    linear_layer = nn.Linear(3, 11, dtype=torch.float32).to(device)
    contextT = contextT.to(torch.float32)
    contextT = linear_layer(contextT)

    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    context = context.to(device)
    attn = attn.to(device)
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

# class Block(nn.Module):
#     class Block(nn.Module):
#         def __init__(
#                 self,
#                 dim=512,
#                 num_heads=8,
#                 mlp_ratio=4.0,
#                 drop_path=0.2,
#                 act_layer=nn.GELU,
#                 norm_layer=LayerNorm,
#                 mlp_layer=nn.Sequential,
#         ):
#             super().__init__()
#
#             self.norm1 = norm_layer(dim)
#             self.attn = nn.MultiheadAttention(dim, num_heads)
#             self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#             self.norm2 = norm_layer(dim)
#             self.mlp = mlp_layer(
#                 nn.Linear(dim, int(dim * mlp_ratio)),
#                 act_layer(),
#                 nn.Dropout(drop_path),
#                 nn.Linear(int(dim * mlp_ratio), dim),
#             )
#             self.drop_path2 = nn.Dropout(drop_path)
#
#     def forward(self, x):
#         # Attention block
#         x_normalized = self.norm1(x)
#         attention_output = self.attn(x_normalized, x_normalized, x_normalized)[0]
#         attention_output_with_dropout = self.drop_path1(attention_output)
#         x = x + attention_output_with_dropout
#
#         # MLP block
#         x = x + self.drop_path2(self.mlp(self.norm2(x)))
#
#         return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int = 512,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = nn.Sequential,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(nn.Linear(dim, int(dim * mlp_ratio)),
                             act_layer(),
                             nn.Dropout(drop_path),
                             nn.Linear(int(dim * mlp_ratio), dim),
                             )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor