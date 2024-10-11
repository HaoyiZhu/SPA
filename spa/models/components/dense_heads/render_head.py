import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.transforms import InterpolationMode

from spa.utils import RankedLogger

from ..model_utils import unet3d_utils
from ..model_utils.render_utils import models as render_models
from ..model_utils.render_utils import rays

log = RankedLogger(__name__, rank_zero_only=True)


class RenderHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        val_ray_split=8192,
        feature_type="3d_to_3d",
        proj_cfg=None,
        render_cfg=None,
        semantic_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.val_ray_split = val_ray_split
        self.feature_type = feature_type
        self.semantic_cfg = semantic_cfg
        self.use_semantic = (
            self.semantic_cfg.get("use_semantic", True)
            if self.semantic_cfg is not None
            else False
        )
        if self.use_semantic:
            self.load_semantic_model()
        proj_cfg = proj_cfg
        self.proj_net = getattr(unet3d_utils, proj_cfg["type"])(
            in_channels=in_channels, **proj_cfg
        )
        render_cfg = render_cfg
        self.renderer = getattr(render_models, render_cfg["type"])(**render_cfg)
        self.forward_ret_dict = {}
        self.freeze_stages()

    def freeze_stages(self):
        if not self.use_semantic:
            return
        elif self.semantic_cfg.type == "radio":
            self.radio.eval()
            self.radio.requires_grad_(False)
        else:
            raise NotImplementedError

    def train(self, mode=True):
        super().train(mode)
        self.freeze_stages()

    @torch.no_grad()
    def load_semantic_model(self):
        assert (
            self.semantic_cfg.type == "radio"
        ), f"Unsupported semantic model type: {self.semantic_cfg.type}"
        if os.path.exists(os.path.expanduser("~/.cache/torch/hub/NVlabs_RADIO_main")):
            self.radio = torch.hub.load(
                os.path.expanduser("~/.cache/torch/hub/NVlabs_RADIO_main"),
                "radio_model",
                version=self.semantic_cfg.img_radio_cfg.model,
                progress=True,
                source="local",
            )
        else:
            self.radio = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=self.semantic_cfg.img_radio_cfg.model,
                progress=True,
                skip_validation=True,
            )
        log.info("Loading pretrained radio model")

    @torch.no_grad()
    def create_imsemantic(self, img):
        img_radio = img
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, spatial_features = self.radio(img_radio)
            ret_features = rearrange(
                spatial_features,
                "b (h w) d -> b d h w",
                h=img_radio.shape[-2] // self.radio.patch_size,
                w=img_radio.shape[-1] // self.radio.patch_size,
            )

        ret_features = ret_features.to(torch.float32)
        return ret_features

    @torch.no_grad()
    def prepare_ray(self, batch_dict):
        bray_p, bray_idx = batch_dict["ray_p"], batch_dict["ray_idx"]
        trans2d_matrix = batch_dict["trans2d_matrix"]
        ori_shape = batch_dict["ori_shape"]
        img = batch_dict["img"]
        img_norm_cfg = batch_dict["img_norm_cfg"]

        if self.use_semantic:
            semantic_img = torch.cat(batch_dict["semantic_img"], dim=0)
            bsemantic = self.create_imsemantic(semantic_img)
            bsemantic = bsemantic.split([len(img[i]) for i in range(len(img))], dim=0)

        ori_img = [
            _img * img_norm_cfg["std"][:, None, None]
            + img_norm_cfg["mean"][:, None, None]
            for _img in img
        ]

        ray_dict = {
            "rgb": [],
            "depth": [d[..., None] for d in batch_dict["ray_depth"]],
            "origin": batch_dict["ray_o"],
            "direction": batch_dict["ray_d"],
            "near": batch_dict["ray_near"],
            "far": batch_dict["ray_far"],
            "scene_scale": batch_dict["ray_scale"],
            "scene_bbox": batch_dict["point_cloud_range"],
        }

        if self.use_semantic:
            ray_dict["semantic"] = []

        for bidx in range(batch_dict["batch_size"]):
            iray_rgb, iray_semantic = [], []
            ray_idx = bray_idx[bidx]
            for cidx in range(len(img[bidx])):
                ray_p = bray_p[bidx][ray_idx == cidx]

                ray_p_rgb = (
                    torch.stack(
                        [
                            ray_p[:, 0] / (img[bidx].shape[-1] - 1),
                            ray_p[:, 1] / (img[bidx].shape[-2] - 1),
                        ],
                        dim=-1,
                    )
                    * 2
                    - 1
                )
                assert torch.all((ray_p_rgb >= -1) & (ray_p_rgb <= 1))
                ray_rgb = (
                    F.grid_sample(
                        ori_img[bidx][cidx : cidx + 1].contiguous(),
                        ray_p_rgb[None, None].contiguous(),
                        align_corners=True,
                    )
                    .squeeze(0)
                    .squeeze(1)
                    .transpose(0, 1)
                )
                iray_rgb.append(ray_rgb)

                if self.use_semantic:
                    shp = ori_shape[bidx][cidx]
                    img2aug = trans2d_matrix[bidx][cidx]
                    ray_p_semantic = torch.stack(
                        [
                            ray_p[:, 0],
                            ray_p[:, 1],
                            torch.ones_like(ray_p[:, 0]),
                            torch.ones_like(ray_p[:, 0]),
                        ],
                        dim=-1,
                    )
                    ray_p_semantic = ray_p_semantic @ torch.linalg.inv(img2aug).T
                    ray_p_semantic = (
                        torch.stack(
                            [
                                ray_p_semantic[:, 0] / (shp[1] - 1),
                                ray_p_semantic[:, 1] / (shp[0] - 1),
                            ],
                            dim=-1,
                        )
                        * 2
                        - 1
                    )
                    assert torch.all((ray_p_semantic >= -1) & (ray_p_semantic <= 1))
                    ray_semantic = (
                        F.grid_sample(
                            bsemantic[bidx][cidx : cidx + 1].contiguous(),
                            ray_p_semantic[None, None].contiguous(),
                            align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(1)
                        .transpose(0, 1)
                    ).contiguous()
                    iray_semantic.append(ray_semantic)

            ray_dict["rgb"].append(torch.cat(iray_rgb, dim=0))
            if self.use_semantic:
                ray_dict["semantic"].append(torch.cat(iray_semantic, dim=0))

        for k in [
            "ray_depth",
            "ray_rgb",
            "ray_o",
            "ray_d",
            "ray_p",
            "ray_scale",
            "ray_near",
            "ray_far",
            "ray_idx",
        ]:
            batch_dict.pop(k, None)

        rearrange_ray_dict = []
        for bidx in range(batch_dict["batch_size"]):
            rearrange_ray_dict.append({k: v[bidx] for k, v in ray_dict.items()})
        return rearrange_ray_dict

    def prepare_volume(self, batch_dict):
        if self.feature_type == "2d_to_3d":
            volume_feat = batch_dict["spatial_features_2d"]
        elif self.feature_type == "3d_to_3d":
            volume_feat = batch_dict["encoded_spconv_tensor"]
        else:
            raise NotImplementedError
        if "dataset_name" in batch_dict.keys():
            dataset = batch_dict["dataset_name"][0]
        else:
            dataset = "default"

        volume_feat = self.proj_net(volume_feat, dataset=dataset)
        return volume_feat

    def render_func(self, ray_dict, volume_feature):
        batched_render_out = []
        for i in range(len(ray_dict)):
            i_ray_o, i_ray_d, i_ray_near, i_ray_far = (
                ray_dict[i]["origin"],
                ray_dict[i]["direction"],
                ray_dict[i]["near"],
                ray_dict[i]["far"],
            )
            i_volume_feature = [v[i] for v in volume_feature]
            i_scene_bbox = ray_dict[i]["scene_bbox"]
            i_scene_scale = ray_dict[i]["scene_scale"]

            if self.training:
                ray_bundle = rays.RayBundle(
                    origins=i_ray_o,
                    directions=i_ray_d,
                    nears=i_ray_near,
                    fars=i_ray_far,
                )
                render_out = self.renderer(
                    ray_bundle, i_volume_feature, i_scene_bbox, i_scene_scale
                )
            else:
                render_out = defaultdict(list)
                for j_ray_o, j_ray_d, j_ray_near, j_ray_far in zip(
                    i_ray_o.split(self.val_ray_split, dim=0),
                    i_ray_d.split(self.val_ray_split, dim=0),
                    i_ray_near.split(self.val_ray_split, dim=0),
                    i_ray_far.split(self.val_ray_split, dim=0),
                ):
                    ray_bundle = rays.RayBundle(
                        origins=j_ray_o,
                        directions=j_ray_d,
                        nears=j_ray_near,
                        fars=j_ray_far,
                    )
                    part_render_out = self.renderer(
                        ray_bundle, i_volume_feature, i_scene_bbox, i_scene_scale
                    )
                    for k, v in part_render_out.items():
                        render_out[k].append(v.detach())
                    del part_render_out
                    torch.cuda.empty_cache()
                for k, v in render_out.items():
                    render_out[k] = torch.cat(v, dim=0)
            batched_render_out.append(render_out)

        return batched_render_out

    def get_loss(self, ray_preds, ray_targets):
        batch_size = len(ray_targets)
        loss_dict = defaultdict(list)
        for bs_idx in range(batch_size):
            i_loss_dict = self.renderer.get_loss(ray_preds[bs_idx], ray_targets[bs_idx])
            for k, v in i_loss_dict.items():
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()
        loss = sum(_value for _key, _value in loss_dict.items() if "loss" in _key)
        return loss, loss_dict

    def forward(self, batch_dict):
        ray_dict = self.prepare_ray(batch_dict)
        volume_feature = self.prepare_volume(batch_dict)
        render_out = self.render_func(ray_dict, volume_feature)
        batch_dict.update({"render_out": render_out, "ray_dict": ray_dict})
        return batch_dict
