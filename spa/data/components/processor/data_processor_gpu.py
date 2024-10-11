from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from . import augmentor_utils


class DataProcessorGPU(object):
    def __init__(self, processor_cfg, mode, logger):
        self.mode = mode
        self.logger = logger
        self.color_jitter = None

        enabled_proc_list = processor_cfg.get("enabled_proc_list", {self.mode: []})
        proc_config = processor_cfg.get("proc_config", {})
        self.data_processor_queue = []
        message = "gpu processor:"
        for proc_name in enabled_proc_list[self.mode]:
            message += f" {proc_name}"
            assert proc_name in proc_config.keys(), f"{proc_name} not in proc_config"
            cur_processor = getattr(self, proc_name)(config=proc_config[proc_name])
            self.data_processor_queue.append(cur_processor)
        # self.logger.info(message)

    def random_photometric_distort(self, batch_dict=None, config=None):
        assert self.mode == "train"
        if batch_dict is None:
            self.color_jitter = augmentor_utils.ColorJitter(
                contrast=config["contrast"],
                saturation=config["saturation"],
                hue=config["hue"],
                brightness=config["brightness"],
                p=config["p"],
            )
            return partial(self.random_photometric_distort, config=config)

        assert config["mv_consistency"]

        img = batch_dict["img"]
        for bs_idx in range(len(img)):
            self.color_jitter.reset_params()
            for cam_idx in range(len(img[bs_idx])):
                img[bs_idx][cam_idx] = self.color_jitter(img[bs_idx][cam_idx])

        batch_dict["img"] = img
        return batch_dict

    def imnormalize(self, batch_dict=None, config=None):
        if batch_dict is None:
            return partial(self.imnormalize, config=config)
        img = batch_dict["img"]
        for bs_idx in range(len(img)):
            mean = img[bs_idx].new_tensor(config["mean"])
            std = img[bs_idx].new_tensor(config["std"])
            img[bs_idx] = (img[bs_idx] - mean[:, None, None]) / std[:, None, None]
        batch_dict["img_norm_cfg"] = {"mean": mean, "std": std}
        return batch_dict

    def filter_depth_outlier(self, batch_dict=None, config=None):
        if batch_dict is None:
            return partial(self.filter_depth_outlier, config=config)

        depth = []
        for bidx in range(len(batch_dict["depth"])):
            i_depth = batch_dict["depth"][bidx].clone()
            i_mask = i_depth > 1e-3
            valid_depth = i_depth[i_mask]
            k = int(valid_depth.numel() * config["percentile"])
            rmax = torch.topk(valid_depth, k, largest=True)[0][-1]
            rmin = torch.topk(valid_depth, k, largest=False)[0][-1]
            i_mask &= (i_depth > rmin) & (i_depth < rmax)
            i_depth[~i_mask] = 0.0
            depth.append(i_depth)

        batch_dict["depth"] = depth
        return batch_dict

    def calc_ray_from_depth(self, batch_dict=None, config=None):
        if batch_dict is None:
            return partial(self.calc_ray_from_depth, config=config)

        cam2img = batch_dict["cam2img"]
        world2cam = batch_dict["world2cam"]
        depth = batch_dict["depth"]
        img = batch_dict["img"]

        (
            batch_ray_depth,
            batch_ray_rgb,
            batch_ray_scale,
            batch_ray_o,
            batch_ray_d,
            batch_ray_p,
        ) = ([], [], [], [], [], [])
        for bidx in range(len(depth)):
            # (N, 3, H, W)
            ray_rgb = img[bidx]
            # (N, 3, H, W) -> (N, H, W, 3)
            ray_rgb = ray_rgb.transpose(-3, -2).transpose(-2, -1).contiguous()

            i_depth = depth[bidx]
            i_mask = i_depth > 1e-3
            i_img2cam = torch.linalg.inv(cam2img[bidx])
            i_cam2world = torch.linalg.inv(world2cam[bidx])
            i_img2world = i_cam2world @ i_img2cam

            assert (i_depth.shape[1] == ray_rgb.shape[1]) and (
                i_depth.shape[2] == ray_rgb.shape[2]
            )
            H, W = i_depth.shape[-2:]
            pixel_y, pixel_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=i_depth.device),
                torch.linspace(0.5, W - 0.5, W, device=i_depth.device),
                indexing="ij",
            )
            # (N, H, W, 4)
            ray_end = torch.stack(
                [
                    pixel_x[None].expand_as(i_depth),
                    pixel_y[None].expand_as(i_depth),
                    torch.where(i_mask, i_depth, torch.ones_like(i_depth)),
                    torch.ones_like(i_depth),
                ],
                dim=-1,
            )
            ray_end[..., :2] *= ray_end[..., 2:3]

            # (N, H, W, 4, 4) @ (N, H, W, 4, 1) -> (N, H, W, 3)
            ray_end = torch.matmul(
                i_img2world[:, None, None, :, :], ray_end[..., None]
            )[..., :3, 0]
            ray_o = i_cam2world[:, None, None, :3, 3].expand_as(ray_end)
            ray_d = ray_end - ray_o
            ray_depth = torch.linalg.norm(ray_d, dim=-1, keepdim=False)
            ray_depth[~i_mask] = 0.0

            ray_scale = torch.clamp(
                ray_depth[i_mask].sum() / torch.clamp(i_mask.sum(), min=1), min=1e-6
            ).item()

            # (H, W, 2) -> (N, H, W, 2)
            ray_p = torch.stack([pixel_x / W, pixel_y / H], dim=-1)
            ray_p = ray_p[None, ...].expand(ray_d[..., :2].shape)

            batch_ray_depth.append(ray_depth.flatten(1, 2).contiguous())
            batch_ray_rgb.append(ray_rgb.flatten(1, 2).contiguous())
            batch_ray_scale.append(ray_scale)
            batch_ray_o.append(ray_o.flatten(1, 2).contiguous())
            batch_ray_d.append(F.normalize(ray_d, dim=-1).flatten(1, 2).contiguous())
            batch_ray_p.append(ray_p.flatten(1, 2).contiguous())

        batch_dict["ray_depth"] = batch_ray_depth
        batch_dict["ray_rgb"] = batch_ray_rgb
        batch_dict["ray_scale"] = batch_ray_scale
        batch_dict["ray_o"] = batch_ray_o
        batch_dict["ray_d"] = batch_ray_d
        batch_dict["ray_p"] = batch_ray_p
        return batch_dict

    def calc_scene_bbox(self, batch_dict=None, config=None):
        if batch_dict is None:
            return partial(self.calc_scene_bbox, config=config)

        point_cloud_range = []
        if config["type"] == "dynamic_depth":
            ray_depth = batch_dict["ray_depth"]
            ray_d = batch_dict["ray_d"]
            ray_o = batch_dict["ray_o"]
            for bidx in range(len(ray_o)):
                i_mask = ray_depth[bidx] > 1e-3
                i_ray_depth = ray_depth[bidx][i_mask]
                i_ray_d = ray_d[bidx][i_mask]
                i_ray_o = ray_o[bidx][i_mask]
                pc = i_ray_o + i_ray_d * i_ray_depth[..., None]
                point_cloud_range.append(
                    torch.cat([pc.min(0).values, pc.max(0).values])
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        elif config["type"] == "dynamic_point":
            raise NotImplementedError
        elif config["type"] == "static":
            point_cloud_range.append(
                np.array(config["point_cloud_range"], dtype=np.float32)
            )
        else:
            raise NotImplementedError

        batch_dict["point_cloud_range"] = point_cloud_range
        return batch_dict

    def calc_voxel_size(self, batch_dict=None, config=None):
        if batch_dict is None:
            return partial(self.calc_voxel_size, config=config)

        point_cloud_range = batch_dict["point_cloud_range"]
        grid_size = []
        voxel_size = []
        for bidx in range(len(point_cloud_range)):
            pcr = point_cloud_range[bidx]
            gs = np.array(config["grid_size"], dtype=np.int64)
            grid_size.append(gs)
            voxel_size.append((pcr[3:] - pcr[:3]) / gs)
        batch_dict["voxel_size"] = voxel_size
        batch_dict["grid_size"] = grid_size
        return batch_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            batch_dict = cur_processor(batch_dict=batch_dict)
        return batch_dict
