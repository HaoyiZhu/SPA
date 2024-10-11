import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_attn.ms_deform_attn_utils import MSDeformAttnFunction
from einops import rearrange
from torch.nn.functional import linear
from torch.nn.parameter import Parameter


def batch_mask_sequence(feats_list, mask):
    """
    Args:
        feats_list: [(B, N, C), ...]
        mask: (B, N)
    Returns:
        rebatch_feats_list: [(B, M, C), ...]
        mask_indices: [(M1,), (M2,), ...]
    """
    batch_size = mask.shape[0]
    mask_indices = []
    for bs_idx in range(batch_size):
        mask_indices.append(mask[bs_idx].nonzero(as_tuple=True)[0])
    max_len = max([len(each) for each in mask_indices])
    rebatch_feats_list = []
    for feats in feats_list:
        rebatch_feats = feats.new_zeros(
            [batch_size, max_len, feats.shape[-1]], dtype=feats.dtype
        )
        for bs_idx in range(batch_size):
            i_index = mask_indices[bs_idx]
            rebatch_feats[bs_idx, : len(i_index)] = feats[bs_idx, i_index]
        rebatch_feats_list.append(rebatch_feats)
    return rebatch_feats_list, mask_indices


def rebatch_mask_sequence(feats, rebatch_feats, mask_indices):
    """
    Args:
        feats: (B, N, C)
        rebatch_feats: (B, M, C)
        mask_indices: [(M1,), (M2,), ...]
    Returns:
        new_feats: (B, N, C)
    """
    batch_size = feats.shape[0]
    new_feats = rebatch_feats.new_zeros(
        [batch_size, feats.shape[1], rebatch_feats.shape[-1]], dtype=rebatch_feats.dtype
    )
    for bs_idx in range(batch_size):
        i_index = mask_indices[bs_idx]
        new_feats[bs_idx, i_index] = rebatch_feats[bs_idx, : len(i_index)]
    return new_feats


class VoxelformerCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_points,
        num_levels,
        im2col_step=64,
        bias=True,
        dropout=0.0,
        **kwargs,
    ):
        super(VoxelformerCrossAttention, self).__init__()
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(
            self.num_heads, dtype=self.sampling_offsets.weight.dtype
        ) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        query_mask=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        length = [tmp.shape[0] for tmp in query_mask]
        B, Z, Y, X, C = query.shape
        query = [
            query[i : i + 1].expand(N, -1, -1, -1, -1) for i, N in enumerate(length)
        ]
        # (N1+N2+..., Z*Y*X, C)
        query = torch.cat(query, dim=0).view(-1, Z * Y * X, C)
        reference_points = torch.cat(reference_points, dim=0).view(-1, Z * Y * X, 2)
        query_mask = torch.cat(query_mask, dim=0).view(-1, Z * Y * X)

        (bvoxel_cam_coords, bvoxel_embeds), bmask_indices = batch_mask_sequence(
            [reference_points, query], query_mask
        )  # [(N1+N2+..., M, 2), (N1+N2+..., M, C)]

        query = bvoxel_embeds
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # (num_level, 2)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )
        # (bs, num_query, num_heads, num_levels, num_points, 2)
        sampling_locations = (
            bvoxel_cam_coords[:, :, None, None, None, :2]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = MSDeformAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = self.out_proj(output)
        output = identity + self.dropout_layer(output)

        output = rebatch_mask_sequence(reference_points, output, bmask_indices)

        presum = 0
        final = []
        for l in length:
            x = output[presum : presum + l].sum(dim=0)
            c = torch.clamp(query_mask[presum : presum + l].sum(dim=0), min=1.0)
            x = (x / c[..., None]).view(Z, Y, X, -1)
            final.append(x)
            presum += l
        # (B, Z, Y, X, C)
        output = torch.stack(final, dim=0)
        return output


class VoxelformerSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_convs,
        **kwargs,
    ):
        super(VoxelformerSelfAttention, self).__init__()
        self.embed_dims = embed_dims
        self.conv_layer = nn.ModuleList()
        for k in range(num_convs):
            self.conv_layer.append(
                nn.Sequential(
                    nn.Conv3d(
                        embed_dims,
                        embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        **kwargs,
    ):
        # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        output = query.permute(0, 4, 1, 2, 3)
        for layer in self.conv_layer:
            output = layer(output)
        # (B, C, Z, Y, X) -> (B, Z, Y, X, C)
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        return output
