from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from .. import fields, ray_samplers
from ..renderers import DepthRenderer, OtherRenderer, RGBRenderer


class SurfaceModel(nn.Module):
    def __init__(
        self,
        field_cfg,
        sampler_cfg,
        loss_cfg,
        **kwargs,
    ):
        super().__init__()
        self.field = getattr(fields, field_cfg["type"])(**field_cfg)
        self.sampler = getattr(ray_samplers, sampler_cfg["type"])(**sampler_cfg)
        self.rgb_renderer = RGBRenderer()
        self.depth_renderer = DepthRenderer()
        self.other_renderer = OtherRenderer()
        self.loss_cfg = loss_cfg

    @abstractmethod
    def sample_and_forward_field(
        self, ray_bundle, volume_feature, scene_bbox, scene_scale
    ):
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_outputs(
        self, ray_bundle, volume_feature, scene_bbox, scene_scale, **kwargs
    ):
        outputs = {}

        samples_and_field_outputs = self.sample_and_forward_field(
            ray_bundle, volume_feature, scene_bbox, scene_scale
        )

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)
        normal = self.other_renderer(vals=field_outputs["normal"], weights=weights)
        if "rgb" in field_outputs.keys():
            rgb = self.rgb_renderer(rgb=field_outputs["rgb"], weights=weights)
            outputs["rgb"] = rgb
        if "semantic" in field_outputs.keys():
            semantic = self.other_renderer(
                vals=field_outputs["semantic"], weights=weights
            )
            outputs["semantic"] = semantic

        outputs.update(
            {
                "depth": depth,
                "normal": normal,
                "weights": weights,
                "sdf": field_outputs["sdf"],
                "gradients": field_outputs["gradients"],
                "z_vals": ray_samples.frustums.starts,
            }
        )

        """ add for visualization"""
        outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        if samples_and_field_outputs.get("init_sampled_points", None) is not None:
            outputs.update(
                {
                    "init_sampled_points": samples_and_field_outputs[
                        "init_sampled_points"
                    ],
                    "init_weights": samples_and_field_outputs["init_weights"],
                    "new_sampled_points": samples_and_field_outputs[
                        "new_sampled_points"
                    ],
                }
            )

        return outputs

    def forward(self, ray_bundle, volume_feature, scene_bbox, scene_scale, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the
        configuration of the model and whether or not the batch is provided (whether or not we are
        training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        ray_bundle.origins /= scene_scale
        ray_bundle.nears /= scene_scale
        ray_bundle.fars /= scene_scale
        return self.get_outputs(
            ray_bundle, volume_feature, scene_bbox, scene_scale, **kwargs
        )

    def get_loss(self, preds_dict, targets):
        loss_dict = {}
        loss_weights = self.loss_cfg.weights
        scene_scale = targets["scene_scale"]

        depth_pred = preds_dict["depth"]  # (num_rays, 1)
        depth_gt = targets["depth"] / scene_scale
        valid_gt_mask = depth_gt > 0.0
        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        # free space loss and sdf loss
        pred_sdf = preds_dict["sdf"][..., 0]
        z_vals = preds_dict["z_vals"][..., 0]
        truncation = self.loss_cfg.sensor_depth_truncation / scene_scale

        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        if loss_weights.get("free_space_loss", 0.0) > 0:
            free_space_loss = (
                F.relu(truncation - pred_sdf) * front_mask
            ).sum() / torch.clamp(front_mask.sum(), min=1.0)
            loss_dict["free_space_loss"] = (
                free_space_loss * loss_weights.free_space_loss
            )

        if loss_weights.get("sdf_loss", 0.0) > 0:
            sdf_loss = (
                torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
            ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
            loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        if loss_weights.get("eikonal_loss", 0.0) > 0:
            gradients = preds_dict["gradients"]
            eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
            loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        if loss_weights.get("rgb_loss", 0.0) > 0:
            rgb_pred = preds_dict["rgb"]  # (num_rays, 3)
            rgb_gt = targets["rgb"]
            rgb_loss = F.l1_loss(rgb_pred, rgb_gt)
            loss_dict["rgb_loss"] = rgb_loss * loss_weights.rgb_loss
            psnr = 20.0 * torch.log10(1.0 / (rgb_pred - rgb_gt).pow(2).mean().sqrt())
            loss_dict["psnr"] = psnr

        if loss_weights.get("semantic_loss", 0.0) > 0:
            semantic_pred = F.normalize(preds_dict["semantic"], dim=-1)
            semantic_gt = F.normalize(targets["semantic"], dim=-1)
            semantic_loss = 1 - (semantic_pred * semantic_gt).sum(-1).mean()
            loss_dict["semantic_loss"] = semantic_loss * loss_weights.semantic_loss

        return loss_dict
