import math

import torch
import torch.nn.functional as F
from grid_sampler import GridSampler3D
from torch import nn

from spa.utils.transforms import components_from_spherical_harmonics

from ..decoders import RGBDecoder, SDFDecoder, SemanticDecoder


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""
        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        density = alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        return density

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS"""

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFFieldExp(nn.Module):
    def __init__(
        self,
        beta_init,
        padding_mode="zeros",
        render_rgb=True,
        render_semantic=False,
        use_alpha=False,
        use_density=False,
        **kwargs
    ):
        super().__init__()
        self.beta_init = beta_init
        self.padding_mode = padding_mode
        self.use_density = use_density
        self.use_alpha = use_alpha
        self.render_rgb = render_rgb
        self.render_semantic = render_semantic
        if use_density:
            # laplace function for transform sdf to density from VolSDF
            self.laplace_density = LaplaceDensity(init_val=self.beta_init)

        if use_alpha:
            # deviation_network to compute alpha from sdf from NeuS
            self.deviation_network = SingleVarianceNetwork(init_val=self.beta_init)

        self._cos_anneal_ratio = 1.0

    def set_cos_anneal_ratio(self, anneal):
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def get_alpha(self, ray_samples, sdf, gradients):
        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self._cos_anneal_ratio)
            + F.relu(-true_cos) * self._cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def feature_sampling(self, pts, volume_feature, scene_bbox, scene_scale, index):
        """
        Args:
            pts: (N, K, 3), [x, y, z], scaled
            feats_volume: (C, Z, Y, X)
        Returns:
            feats: (N, K, C)
        """
        scene_bbox = pts.new_tensor(scene_bbox)
        pts_norm = (pts * scene_scale - scene_bbox[:3]) / (
            scene_bbox[3:] - scene_bbox[:3]
        )
        pts_norm = pts_norm * 2 - 1  # [0, 1] -> [-1, 1]

        ret_feat = (
            GridSampler3D.apply(
                volume_feature[index].unsqueeze(0).contiguous().to(pts_norm.dtype),
                pts_norm[None, None].contiguous(),
                self.padding_mode,
                True,
            )
            .squeeze(0)
            .squeeze(1)
            .permute(1, 2, 0)
            .contiguous()
        )
        # (1, C, 1, N, K) -> (N, K, C)

        return ret_feat

    def get_sdf(self, points, volume_feature, scene_bbox, scene_scale):
        """predict the sdf value for ray samples"""
        sdf = self.feature_sampling(
            points, volume_feature, scene_bbox, scene_scale, index=0
        )
        return (sdf,)

    def get_density(self, ray_samples, volume_feature, scene_bbox, scene_scale):
        """Computes and returns the densities."""
        points = ray_samples.frustums.get_positions()
        sdf = self.get_sdf(points, volume_feature, scene_bbox, scene_scale)[0]
        density = self.laplace_density(sdf)
        return density

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = torch.sigmoid(-10.0 * sdf)
        return occupancy

    def forward(self, ray_samples, volume_feature, scene_bbox, scene_scale):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}

        points = ray_samples.frustums.get_positions()  # (num_rays, num_samples, 3)

        points.requires_grad_(True)
        with torch.enable_grad():
            (sdf,) = self.get_sdf(points, volume_feature, scene_bbox, scene_scale)

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if self.render_rgb:
            directions = ray_samples.frustums.directions  # (num_rays, num_samples, 3)

            sh = self.feature_sampling(
                points, volume_feature, scene_bbox, scene_scale, index=1
            )
            sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)

            levels = int(math.sqrt(sh.shape[-1]))
            components = components_from_spherical_harmonics(
                levels=levels, directions=directions
            )

            rgb = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
            rgb = torch.sum(sh, dim=-1) + 0.5  # [..., num_samples, 3]
            rgb = torch.sigmoid(rgb)

            outputs["rgb"] = rgb

        if self.render_semantic:
            semantic = self.feature_sampling(
                points, volume_feature, scene_bbox, scene_scale, index=-1
            )
            outputs["semantic"] = semantic

        outputs.update(
            {
                "sdf": sdf,
                "gradients": gradients,
                "normal": F.normalize(gradients, dim=-1),  # TODO: should normalize?
            }
        )

        if self.use_density:
            density = self.laplace_density(sdf)
            outputs["density"] = density

        if self.use_alpha:
            # TODO use mid point sdf for NeuS
            # (num_rays, num_samples, 1)
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs["alphas"] = alphas

        return outputs
