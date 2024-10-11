import os
import re

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spa.utils as U
from spa.data.components.processor.data_processor_gpu import DataProcessorGPU

logger = U.RankedLogger(__name__, rank_zero_only=True)


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


class SPA(nn.Module):
    def __init__(
        self,
        fp16_enabled_layers=[],
        img_backbone=None,
        view_transform=None,
        dense_head=None,
        data_processor_cfg={},
        ckpt_name=None,
    ):
        super().__init__()

        self.logger = logger

        self.fp16_enabled_layers = fp16_enabled_layers
        self.img_backbone = img_backbone
        self.view_transform = view_transform
        self.dense_head = dense_head
        self.data_processor_cfg = data_processor_cfg
        self.data_processor = None
        self.init_data_processor()

        if ckpt_name is not None:
            self.load_pretrained(ckpt_name)

        self.logger.info("----------- FP16 Enabled Status -----------")
        for module_name in self.fp16_enabled_layers:
            getattr(self, module_name).fp16_enabled = True
            self.logger.info(
                f"{module_name}: {getattr(self, module_name).fp16_enabled}"
            )

    def load_pretrained(self, ckpt_name: str = None):
        assert ckpt_name in [
            "spa-l",
            "spa-b",
        ], f"`ckpt_name` should be 'spa-l' or 'spa-b', got {ckpt_name}"

        from huggingface_hub import hf_hub_download

        try:
            import safetensors.torch

            _has_safetensors = True
        except ImportError:
            _has_safetensors = False

        if _has_safetensors:
            from safetensors.torch import load_file

            ckpt_file = hf_hub_download(
                repo_id="HaoyiZhu/SPA", filename=f"{ckpt_name}.safetensors"
            )
            state_dict = load_file(ckpt_file)
        else:
            ckpt_file = hf_hub_download(
                repo_id="HaoyiZhu/SPA", filename=f"{ckpt_name}.ckpt"
            )
            state_dict = torch.load(ckpt_file)["state_dict"]

        self.load_state_dict(state_dict, strict=True)

    def init_data_processor(self):
        self.data_processor = DataProcessorGPU(
            self.data_processor_cfg, mode=self.mode, logger=self.logger
        )

    @property
    def mode(self):
        return "train" if self.training else "test"

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self.init_data_processor()

    @torch.amp.autocast(
        "cuda", enabled=False
    )  # manually enable half precision in each module
    def forward(self, batch_dict):
        batch_dict = self.data_processor.forward(batch_dict=batch_dict)
        batch_dict = self.dense_head(self.view_transform(self.img_backbone(batch_dict)))

        render_out = batch_dict.pop("render_out")
        ray_dict = batch_dict.pop("ray_dict")
        loss, loss_dict = self.dense_head.get_loss(render_out, ray_dict)

        out_dict = dict(loss=loss, **loss_dict)

        if not self.training:
            out_dict.update(self.prepare_visualize(render_out, batch_dict, ray_dict))

        return out_dict

    @torch.no_grad()
    def prepare_visualize(self, render_out, data_dict, ray_dict):
        W, H = (
            int(data_dict["depth"][0].shape[-1]),
            int(data_dict["depth"][0].shape[-2]),
        )

        gt_img = ray_dict[0]["rgb"].reshape(-1, W, 3) * 255.0
        pred_img = render_out[0]["rgb"].reshape(-1, W, 3) * 255.0

        gt_depth = ray_dict[0]["depth"].reshape(-1, W)
        pred_depth = render_out[0]["depth"].reshape(-1, W)

        pred_normal = (
            render_out[0]["normal"].float().reshape(-1, W, 3) * 127.5 + 127.5
        ).clip(0, 255)

        if "masked_inputs" in data_dict and data_dict["masked_inputs"] is not None:
            img_norm_cfg = self.model_cfg.img_norm_cfg
            masked_inputs = einops.rearrange(
                data_dict["masked_inputs"], "n c h w -> (n h) w c", c=3, w=W
            )
            mask = 1.0 - (masked_inputs == 0).float()
            masked_inputs = (
                (
                    masked_inputs
                    * torch.FloatTensor(img_norm_cfg["std"])
                    .to(masked_inputs.device)
                    .reshape(1, 1, 3)
                    + torch.FloatTensor(img_norm_cfg["mean"])
                    .to(masked_inputs.device)
                    .reshape(1, 1, 3)
                )
                * mask
                * 255.0
            )

            if "mae_pred" in data_dict:
                mae_pred = einops.rearrange(
                    data_dict["mae_pred"], "n c h w -> (n h) w c", c=3, w=W
                )
                mae_pred = (
                    mae_pred
                    * torch.FloatTensor(img_norm_cfg["std"])
                    .to(mae_pred.device)
                    .reshape(1, 1, 3)
                    + torch.FloatTensor(img_norm_cfg["mean"])
                    .to(mae_pred.device)
                    .reshape(1, 1, 3)
                ) * 255.0
                mask = einops.repeat(data_dict["mask"], "n 1 h w -> (n h) w 3", w=W)
                paste_img = masked_inputs * (1 - mask) + mae_pred * mask
                gt_img = torch.cat([gt_img, masked_inputs, mae_pred, paste_img], dim=1)
            else:
                gt_img = torch.cat([gt_img, masked_inputs], dim=1)

        if "semantic" in render_out[0]:
            semantic_pred = render_out[0]["semantic"]
            semantic_gt = ray_dict[0]["semantic"]
            similarity = torch.cosine_similarity(
                F.normalize(semantic_pred, dim=-1).detach().clone(),
                F.normalize(semantic_gt, dim=-1).detach().clone(),
                dim=-1,
            ).reshape(-1, W)
            semantic_gt_pca = get_pca_map(
                einops.rearrange(
                    semantic_gt,
                    "(b h w) c -> 1 (b h) w c",
                    h=H,
                    w=W,
                    c=semantic_gt.shape[-1],
                )
            )
            semantic_pred_pca = get_pca_map(
                einops.rearrange(
                    semantic_pred,
                    "(b h w) c -> 1 (b h) w c",
                    h=H,
                    w=W,
                    c=semantic_pred.shape[-1],
                )
            )
            return dict(
                gt_img=gt_img.cpu().numpy(),
                pred_img=pred_img.cpu().numpy(),
                gt_depth=gt_depth.cpu().numpy(),
                pred_depth=pred_depth.cpu().numpy(),
                pred_normal=pred_normal.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                semantic_gt_pca=semantic_gt_pca,
                semantic_pred_pca=semantic_pred_pca,
            )

        return dict(
            gt_img=gt_img.cpu().numpy(),
            pred_img=pred_img.cpu().numpy(),
            gt_depth=gt_depth.cpu().numpy(),
            pred_depth=pred_depth.cpu().numpy(),
            pred_normal=pred_normal.cpu().numpy(),
        )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^img_backbone.cls_token|img_backbone.pos_embed|img_backbone.patch_embed",  # stem and embed
            blocks=[
                (r"^img_backbone.blocks\.(\d+)", None),
                (r"^img_backbone.norm", (99999,)),
            ],
        )
