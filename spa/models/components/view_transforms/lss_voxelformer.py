import numpy as np
import torch
from torch import nn

from ..model_utils.transformer_utils import TransformerLayerSequence


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_channel, embed_dims=256):
        super().__init__()
        self.position_embedding = nn.Sequential(
            nn.Linear(input_channel, embed_dims),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        position_embedding = self.position_embedding(xyz)
        return position_embedding


class LSSVoxelformer(nn.Module):
    def __init__(
        self, in_channels, grid_size, feature_map_stride, transformer_cfg, **kwargs
    ):
        super().__init__()
        self.grid_size = np.array(grid_size, dtype=np.int64) // feature_map_stride
        self.feature_map_stride = feature_map_stride
        self.in_channels = in_channels
        self.register_buffer("voxels", self.create_voxels())
        self.position_encoding = LearnablePositionalEncoding(
            3, embed_dims=self.in_channels
        )
        self.decoder = TransformerLayerSequence(**transformer_cfg)

    def create_voxels(self):
        zs, ys, xs = torch.meshgrid(
            *[torch.arange(0, gs, dtype=torch.float) for gs in self.grid_size[::-1]],
            indexing="ij"
        )
        # [0, grid_size-1]
        voxels = torch.stack([xs, ys, zs], dim=-1)  # (Z, Y, X, 3)
        return voxels

    def prepare_voxels(self, voxel_size, point_cloud_range):
        voxel_coords = self.voxels
        voxel_size = voxel_coords.new_tensor(voxel_size)
        point_cloud_range = voxel_coords.new_tensor(point_cloud_range)
        voxel_coords = (voxel_coords + 0.5) * voxel_size

        voxel_embeds = voxel_coords / (point_cloud_range[3:6] - point_cloud_range[:3])
        assert torch.all((voxel_embeds < 1.0) & (voxel_embeds > 0.0))
        Z, Y, X = voxel_embeds.shape[:3]
        # (Z, Y, X, C)
        voxel_embeds = self.position_encoding(voxel_embeds.view(Z * Y * X, -1))
        voxel_embeds = voxel_embeds.view(Z, Y, X, -1)

        voxel_coords = voxel_coords + point_cloud_range[:3]
        return voxel_coords, voxel_embeds

    def transform_voxels(self, voxel_coords, img, world2cam, cam2img):
        # (Z, Y, X, 4)
        voxel_coords = torch.cat(
            [voxel_coords, torch.ones_like(voxel_coords[..., :1])], dim=-1
        )

        # (N, 4, 4)
        world2img = cam2img @ world2cam

        # (N, 1, 1, 1, 4, 4) @ (1, Z, Y, X, 4, 1) -> (N, Z, Y, X, 3)
        voxel_cam_coords = (
            world2img[:, None, None, None] @ voxel_coords[None, ..., None]
        )[..., :3, 0]

        eps = 1e-5
        voxel_cam_depths = voxel_cam_coords[..., 2].clone()
        mask = voxel_cam_depths > eps
        # (N, Z, Y, X, 2)
        voxel_cam_coords = voxel_cam_coords[..., :2] / torch.maximum(
            voxel_cam_depths, torch.ones_like(voxel_cam_depths) * eps
        ).unsqueeze(-1)

        H, W = img.shape[-2:]
        voxel_cam_coords[..., 0] /= W
        voxel_cam_coords[..., 1] /= H

        mask &= (
            (voxel_cam_coords[..., 0] > 0)
            & (voxel_cam_coords[..., 0] < 1)
            & (voxel_cam_coords[..., 1] > 0)
            & (voxel_cam_coords[..., 1] < 1)
        )

        return voxel_cam_coords, voxel_cam_depths, mask

    def inner_forward(self, mlvl_feats, voxel_cam_coords, voxel_embeds, mask):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            _, _, H, W = feat.shape
            spatial_shape = (H, W)
            # (N1+N2+..., C, H, W) -> (N1+N2+..., C, H*W) -> (N1+N2+..., H*W, C)
            feat = feat.flatten(-2).permute(0, 2, 1).contiguous()
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # (N1+N2+..., H1*W1+H2*W2+..., C)
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )  # (num_level, 2)
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
            ],
            dim=0,
        )  # (num_level,), [0, H1*W1, H1*W1+H2*W2, ...]

        voxel_features = self.decoder(
            query=voxel_embeds,
            value=feat_flatten,
            key=feat_flatten,
            reference_points=voxel_cam_coords,
            query_mask=mask,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
        )
        voxel_features = voxel_features.permute(0, 4, 1, 2, 3).contiguous()
        return voxel_features

    def forward(self, batch_dict):
        # [(N1+N2+..., C, H, W), ...]
        img_features = batch_dict["img_features"]
        batch_size = batch_dict["batch_size"]
        world2cam = batch_dict["world2cam"]
        cam2img = batch_dict["cam2img"]
        img = batch_dict["img"]
        voxel_size = batch_dict["voxel_size"]
        point_cloud_range = batch_dict["point_cloud_range"]

        voxel_embeds, voxel_cam_coords, mask = (
            [],
            [],
            [],
        )
        for bidx in range(batch_size):
            vs = voxel_size[bidx] * self.feature_map_stride
            pcr = point_cloud_range[bidx]
            vc, ve = self.prepare_voxels(vs, pcr)

            l2c = world2cam[bidx]
            c2i = cam2img[bidx]
            vcc, _, m = self.transform_voxels(vc, img[bidx], l2c, c2i)

            voxel_embeds.append(ve)
            voxel_cam_coords.append(vcc)
            mask.append(m)
        # (B, Z, Y, X, C)
        voxel_embeds = torch.stack(voxel_embeds, dim=0)

        # (B, C, Z, Y, X)
        encoded_spconv_tensor = self.inner_forward(
            img_features, voxel_cam_coords, voxel_embeds, mask
        )
        batch_dict["encoded_spconv_tensor"] = encoded_spconv_tensor
        batch_dict["encoded_spconv_tensor_stride"] = self.feature_map_stride
        return batch_dict
