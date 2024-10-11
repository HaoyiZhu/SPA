import copy
from functools import partial, reduce

import cv2
import numpy as np
import torch

from spa.models.components.model_utils.render_utils import scene_colliders

from . import augmentor_utils


class DataProcessor(object):
    def __init__(self, processor_cfg, mode, logger):
        self.mode = mode
        self.logger = logger
        self.collider = None
        enabled_proc_list = processor_cfg.get("enabled_proc_list", {self.mode: []})
        print(f"Init {self.mode} DataProcessor with {enabled_proc_list}")
        proc_config = processor_cfg.get("proc_config", {})
        self.data_processor_queue = []
        for proc_name in enabled_proc_list[self.mode]:
            assert proc_name in proc_config.keys(), f"{proc_name} not in proc_config"
            cur_processor = getattr(self, proc_name)(config=proc_config[proc_name])
            self.data_processor_queue.append(cur_processor)

    def random_world_drop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_drop, config=config)

        points = data_dict["points"]
        points = getattr(augmentor_utils, "global_drop")(
            points, config["drop_ratio"], config["probability"]
        )

        data_dict["points"] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes = data_dict.get("gt_boxes", None)
        points = data_dict.get("points", None)
        matrix = []
        for cur_axis in config["along_axis_list"]:
            assert cur_axis in ["x", "y"]
            gt_boxes, points, mat = getattr(
                augmentor_utils, "random_flip_along_%s" % cur_axis
            )(gt_boxes, points, config["probability"])
            matrix.append(mat)
        matrix = reduce(np.dot, matrix[::-1])

        if gt_boxes is not None:
            data_dict["gt_boxes"] = gt_boxes
        if points is not None:
            data_dict["points"] = points
        data_dict["trans3d_matrix"] = matrix @ data_dict["trans3d_matrix"]
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        gt_boxes = data_dict.get("gt_boxes", None)
        points = data_dict.get("points", None)
        rot_range = config["world_rot_angle"]
        gt_boxes, points, matrix = augmentor_utils.global_rotation(
            gt_boxes, points, rot_range, config["probability"]
        )

        if gt_boxes is not None:
            data_dict["gt_boxes"] = gt_boxes
        if points is not None:
            data_dict["points"] = points
        data_dict["trans3d_matrix"] = matrix @ data_dict["trans3d_matrix"]
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes = data_dict.get("gt_boxes", None)
        points = data_dict.get("points", None)
        gt_boxes, points, matrix = augmentor_utils.global_scaling(
            gt_boxes,
            points,
            config["world_scale_range"],
            config["probability"],
        )

        if gt_boxes is not None:
            data_dict["gt_boxes"] = gt_boxes
        if points is not None:
            data_dict["points"] = points
        data_dict["trans3d_matrix"] = matrix @ data_dict["trans3d_matrix"]
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)

        gt_boxes = data_dict.get("gt_boxes", None)
        points = data_dict.get("points", None)
        noise_translate_std = config["noise_translate_std"]
        assert len(noise_translate_std) == 3
        gt_boxes, points, matrix = augmentor_utils.global_translation(
            gt_boxes,
            points,
            noise_translate_std,
            config["probability"],
        )

        if gt_boxes is not None:
            data_dict["gt_boxes"] = gt_boxes
        if points is not None:
            data_dict["points"] = points
        data_dict["trans3d_matrix"] = matrix @ data_dict["trans3d_matrix"]
        return data_dict

    def filter_depth_outlier(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_depth_outlier, config=config)

        new_depth = []
        for i, _depth in enumerate(data_dict["depth"]):
            mask = _depth > 1e-3
            valid_depth = _depth[mask]
            k = int(len(valid_depth) * config["percentile"])
            dmax = np.sort(np.partition(valid_depth, -k)[-k:])[0]
            dmin = np.sort(np.partition(valid_depth, k)[:k])[-1]
            mask &= (_depth > dmin) & (_depth < dmax)
            new_depth.append(_depth * mask)

        data_dict["depth"] = new_depth
        return data_dict

    def imresize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imresize, config=config)

        extra_keys = config.get("extra_keys", [])
        extra_imgs = {key: [] for key in extra_keys}
        new_img = []
        new_trans2d_matrix = []
        resize = None
        for i, _img in enumerate(data_dict["img"]):
            if not config["mv_consistency"] or resize is None:
                resize = np.random.uniform(*config["resize_scale"][self.mode])
            h, w = _img.shape[:2]
            new_size = (int(w * resize), int(h * resize))
            new_img.append(augmentor_utils.resize(_img, new_size, "lanczos", "pillow"))
            matrix = np.eye(4)
            matrix[0, 0] = new_size[0] / w
            matrix[1, 1] = new_size[1] / h
            new_trans2d_matrix.append(matrix @ data_dict["trans2d_matrix"][i])

            for extra_key in extra_keys:
                if extra_key not in data_dict.keys():
                    continue
                extra_img = data_dict[extra_key][i]
                assert extra_img.shape[:2] == _img.shape[:2]
                extra_imgs[extra_key].append(
                    augmentor_utils.resize(extra_img, new_size, "nearest", "cv2")
                )

        data_dict["img"] = new_img
        data_dict["trans2d_matrix"] = new_trans2d_matrix
        for extra_key in extra_keys:
            if extra_key not in data_dict.keys():
                continue
            data_dict[extra_key] = extra_imgs[extra_key]
        return data_dict

    def imcrop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imcrop, config=config)

        extra_keys = config.get("extra_keys", [])
        extra_imgs = {key: [] for key in extra_keys}
        new_img = []
        new_trans2d_matrix = []
        crop_pos_w, crop_pos_h = None, None
        for i, _img in enumerate(data_dict["img"]):
            crop_size = config["crop_size"]
            if not config["mv_consistency"] or crop_pos_w is None:
                crop_pos_h = np.random.uniform(*config["crop_pos"][self.mode][0])
                crop_pos_w = np.random.uniform(*config["crop_pos"][self.mode][1])

            start_h = int(crop_pos_h * max(0, _img.shape[0] - crop_size[0]))
            start_w = int(crop_pos_w * max(0, _img.shape[1] - crop_size[1]))

            _img_src = augmentor_utils.crop(
                _img, start_h, start_w, crop_size[0], crop_size[1]
            )
            new_img.append(_img_src)
            matrix = np.eye(4)
            matrix[0, 2] = -start_w
            matrix[1, 2] = -start_h
            new_trans2d_matrix.append(matrix @ data_dict["trans2d_matrix"][i])

            for extra_key in extra_keys:
                if extra_key not in data_dict.keys():
                    continue
                extra_img = data_dict[extra_key][i]
                assert extra_img.shape[:2] == _img.shape[:2]
                _extra_img_src = augmentor_utils.crop(
                    extra_img, start_h, start_w, crop_size[0], crop_size[1]
                )
                extra_imgs[extra_key].append(_extra_img_src)

        data_dict["img"] = new_img
        data_dict["trans2d_matrix"] = new_trans2d_matrix
        for extra_key in extra_keys:
            if extra_key not in data_dict.keys():
                continue
            data_dict[extra_key] = extra_imgs[extra_key]
        return data_dict

    def imflip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imflip, config=config)

        extra_keys = config.get("extra_keys", [])
        extra_imgs = {key: [] for key in extra_keys}
        new_img = []
        new_trans2d_matrix = []
        enable = None
        for i, _img in enumerate(data_dict["img"]):
            if not config["mv_consistency"] or enable is None:
                flip_ratio = config["flip_ratio"]
                enable = np.random.choice(
                    [False, True], replace=False, p=[1 - flip_ratio, flip_ratio]
                )
            matrix = np.eye(4)
            if enable:
                _img = np.flip(_img, axis=1)
                matrix[0, 0] = -1
                matrix[0, 2] = _img.shape[1] - 1
            new_img.append(_img)
            new_trans2d_matrix.append(matrix @ data_dict["trans2d_matrix"][i])

            for extra_key in extra_keys:
                if extra_key not in data_dict.keys():
                    continue
                extra_img = data_dict[extra_key][i]
                assert extra_img.shape[:2] == _img.shape[:2]
                extra_imgs[extra_key].append(
                    np.flip(extra_img, axis=1) if enable else extra_img
                )

        data_dict["img"] = new_img
        data_dict["trans2d_matrix"] = new_trans2d_matrix
        for extra_key in extra_keys:
            if extra_key not in data_dict.keys():
                continue
            data_dict[extra_key] = extra_imgs[extra_key]
        return data_dict

    def imrotate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imrotate, config=config)

        extra_keys = config.get("extra_keys", [])
        extra_imgs = {key: [] for key in extra_keys}
        new_img = []
        new_trans2d_matrix = []
        angle = None
        for i, _img in enumerate(data_dict["img"]):
            if not config["mv_consistency"] or angle is None:
                angle = np.random.uniform(*config["rotate_angle"])
            h, w = _img.shape[:2]
            c_x, c_y = (w - 1) * 0.5, (h - 1) * 0.5
            matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1)
            new_img.append(cv2.warpAffine(_img, matrix, (w, h)))
            rot_sin, rot_cos = np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)
            matrix = np.eye(4)
            matrix[:2, :3] = np.array(
                [
                    [rot_cos, -rot_sin, (1 - rot_cos) * c_x + rot_sin * c_y],
                    [rot_sin, rot_cos, (1 - rot_cos) * c_y - rot_sin * c_x],
                ]
            )
            new_trans2d_matrix.append(matrix @ data_dict["trans2d_matrix"][i])

            for extra_key in extra_keys:
                if extra_key not in data_dict.keys():
                    continue
                extra_img = data_dict[extra_key][i]
                assert extra_img.shape[:2] == _img.shape[:2]
                extra_imgs[extra_key].append(
                    cv2.warpAffine(extra_img, matrix, (w, h), flags=cv2.INTER_NEAREST)
                )

        data_dict["img"] = new_img
        data_dict["trans2d_matrix"] = new_trans2d_matrix
        for extra_key in extra_keys:
            if extra_key not in data_dict.keys():
                continue
            data_dict[extra_key] = extra_imgs[extra_key]
        return data_dict

    def imnormalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imnormalize, config=config)
        new_img = []
        mean = np.array(config["mean"], dtype=np.float32)
        std = np.array(config["std"], dtype=np.float32)
        to_rgb = config.get("to_rgb", False)
        for i, _img in enumerate(data_dict["img"]):
            _img = _img.astype(np.float32)
            if to_rgb:
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            _img = (_img - mean) / std
            new_img.append(_img)
        data_dict["img"] = new_img
        data_dict["img_norm_cfg"] = {"to_rgb": to_rgb, "mean": mean, "std": std}
        return data_dict

    def impad(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.impad, config=config)

        extra_keys = config.get("extra_keys", [])
        extra_imgs = {key: [] for key in extra_keys}
        new_img = []
        for i, _img in enumerate(data_dict["img"]):
            if config.get("size", None) is not None:
                size = config["size"]
            else:
                size_divisor = config["size_divisor"]
                size = (
                    int(np.ceil(_img.shape[0] / size_divisor)) * size_divisor,
                    int(np.ceil(_img.shape[1] / size_divisor)) * size_divisor,
                )
            padding = (0, 0, size[1] - _img.shape[1], size[0] - _img.shape[0])
            _img = cv2.copyMakeBorder(
                _img,
                padding[1],
                padding[3],
                padding[0],
                padding[2],
                cv2.BORDER_CONSTANT,
                value=0,
            )
            new_img.append(_img)

            for extra_key in extra_keys:
                if extra_key not in data_dict.keys():
                    continue
                extra_img = data_dict[extra_key][i]
                assert extra_img.shape[:2] == _img.shape[:2]
                extra_imgs[extra_key].append(
                    cv2.copyMakeBorder(
                        extra_img,
                        padding[1],
                        padding[3],
                        padding[0],
                        padding[2],
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                )

        data_dict["img"] = new_img
        for extra_key in extra_keys:
            if extra_key not in data_dict.keys():
                continue
            data_dict[extra_key] = extra_imgs[extra_key]
        return data_dict

    def grid_mask(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.grid_mask, config=config)

        new_img = []
        mask, mask_noise = None, None
        for i, _img in enumerate(data_dict["img"]):
            if not config["mv_consistency"] or mask is None:
                mask, mask_noise = augmentor_utils.create_grid_mask(
                    _img.shape,
                    mask_ratio=config["mask_ratio"],
                    probability=config["probability"],
                    rotate_angle=config["rotate_angle"],
                    add_noise=config["add_noise"],
                    rise_with_epoch=config["rise_with_epoch"],
                    epoch_state=data_dict["epoch_state"],
                )
            new_img.append(_img * mask[..., None] + mask_noise[..., None])
        data_dict["img"] = new_img
        return data_dict

    def trans_to_local(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.trans_to_local, config=config)
        for key, val in config["mapping_keys"].items():
            data_dict[val] = data_dict.pop(key)
        if config.get("to_local_key", None) is not None:
            key, inverse = config["to_local_key"]
            # We assume the matrix shape is: (N, 4, 4)
            inv_mat = np.linalg.inv(data_dict[key][0])
            data_dict[key] = (
                data_dict[key] @ inv_mat[None]
                if inverse
                else inv_mat[None] @ data_dict[key]
            )

        return data_dict

    def merge_trans_matrix(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.merge_trans_matrix, config=config)

        if "trans2d_matrix" in data_dict.keys():
            data_dict["trans2d_matrix"] = np.stack(data_dict["trans2d_matrix"], axis=0)

        for key, val in config["keys"].items():
            for data_key, inverse in val:
                trans_matrix = (
                    np.linalg.inv(data_dict[key]) if inverse else data_dict[key]
                )
                data_dict[data_key] = (
                    data_dict[data_key] @ trans_matrix
                    if inverse
                    else trans_matrix @ data_dict[data_key]
                )

        return data_dict

    def filter_depth_outlier_old(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_depth_outlier_old, config=config)

        depth = np.stack(data_dict["depth"], axis=0)
        mask = depth > 1e-3
        valid_depth = depth[mask]
        k = int(len(valid_depth) * config["percentile"])
        dmax = np.sort(np.partition(valid_depth, -k)[-k:])[0]
        dmin = np.sort(np.partition(valid_depth, k)[:k])[-1]
        mask &= (depth > dmin) & (depth < dmax)

        new_depth = []
        for i in range(len(depth)):
            new_depth.append(depth[i] * mask[i])

        data_dict["depth"] = new_depth
        return data_dict

    def calc_ray_from_depth(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.calc_ray_from_depth, config=config)

        # (N, 4, 4)
        cam2img = data_dict["cam2img"]
        world2cam = data_dict["world2cam"]
        # (N, H, W)
        depth = np.stack(data_dict["depth"], axis=0)
        N = len(depth)

        mask = depth > 1e-3
        img2cam = np.linalg.inv(cam2img)
        cam2world = np.linalg.inv(world2cam)
        img2world = cam2world @ img2cam

        H, W = depth.shape[-2:]
        pixel_y, pixel_x = np.meshgrid(
            np.linspace(0.0, H - 1.0, H),
            np.linspace(0.0, W - 1.0, W),
            indexing="ij",
        )
        ray_end = np.stack(
            [
                np.broadcast_to(pixel_x, depth.shape),
                np.broadcast_to(pixel_y, depth.shape),
                np.where(mask, depth, np.ones_like(depth)),
                np.ones_like(depth),
            ],
            axis=-1,
        )
        ray_end[..., :2] *= ray_end[..., 2:3]

        # (N, H, W, 4, 4) @ (N, H, W, 4, 1) -> (N, H, W, 3)
        ray_end = np.matmul(img2world[:, None, None, :, :], ray_end[..., None])[
            ..., :3, 0
        ]
        ray_o = np.broadcast_to(cam2world[:, None, None, :3, 3], ray_end.shape)
        ray_d = ray_end - ray_o
        ray_d_unit = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_depth = np.linalg.norm(ray_d, axis=-1)
        ray_depth[~mask] = 0.0

        ray_scale = max(ray_depth[mask].sum() / max(mask.sum(), 1), 1e-6)

        ray_p = np.stack([pixel_x / W, pixel_y / H], axis=-1)
        ray_p = np.broadcast_to(ray_p[None, ...], ray_d[..., :2].shape)

        data_dict["ray_depth"] = ray_depth.reshape(N, -1)
        data_dict["ray_scale"] = ray_scale
        data_dict["ray_o"] = ray_o.reshape(N, -1, 3)
        data_dict["ray_d"] = ray_d_unit.reshape(N, -1, 3)
        data_dict["ray_p"] = ray_p.reshape(N, -1, 2)
        return data_dict

    def calc_ray_from_depth_v2(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.calc_ray_from_depth_v2, config=config)

        # (N, 4, 4)
        cam2img = data_dict["cam2img"]
        world2cam = data_dict["world2cam"]
        # (N, H, W)
        depth = np.stack(data_dict["depth"], axis=0)
        N = len(depth)

        mask = depth > 1e-3
        img2cam = np.linalg.inv(cam2img)
        cam2world = np.linalg.inv(world2cam)
        img2world = cam2world @ img2cam

        H, W = depth.shape[-2:]
        pixel_y, pixel_x = np.meshgrid(
            np.linspace(0.0, H - 1.0, H),
            np.linspace(0.0, W - 1.0, W),
            indexing="ij",
        )
        ray_end = np.stack(
            [
                np.broadcast_to(pixel_x, depth.shape),
                np.broadcast_to(pixel_y, depth.shape),
                np.where(mask, depth, np.ones_like(depth)),
                np.ones_like(depth),
            ],
            axis=-1,
        )
        ray_end[..., :2] *= ray_end[..., 2:3]

        # (N, H, W, 4, 4) @ (N, H, W, 4, 1) -> (N, H, W, 3)
        ray_end = np.matmul(img2world[:, None, None, :, :], ray_end[..., None])[
            ..., :3, 0
        ]
        ray_o = np.broadcast_to(cam2world[:, None, None, :3, 3], ray_end.shape)
        ray_d = ray_end - ray_o
        ray_d_unit = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_depth = np.linalg.norm(ray_d, axis=-1)
        ray_depth[~mask] = 0.0

        ray_scale = max(ray_depth[mask].sum() / max(mask.sum(), 1), 1e-6)

        ray_p = np.stack([pixel_x, pixel_y], axis=-1)
        ray_p = np.broadcast_to(ray_p[None, ...], ray_d[..., :2].shape)

        data_dict["ray_depth"] = ray_depth.reshape(N, -1)
        data_dict["ray_scale"] = ray_scale
        data_dict["ray_o"] = ray_o.reshape(N, -1, 3)
        data_dict["ray_d"] = ray_d_unit.reshape(N, -1, 3)
        data_dict["ray_p"] = ray_p.reshape(N, -1, 2)

        return data_dict

    def calc_scene_bbox(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.calc_scene_bbox, config=config)

        if config["type"] == "dynamic_depth":
            ray_depth = data_dict["ray_depth"]
            ray_d = data_dict["ray_d"]
            ray_o = data_dict["ray_o"]
            mask = ray_depth > 1e-3
            ray_depth = ray_depth[mask]
            ray_d = ray_d[mask]
            ray_o = ray_o[mask]
            pc = ray_o + ray_d * ray_depth[..., None]
            point_cloud_range = np.concatenate([pc.min(axis=0), pc.max(axis=0)])
        elif config["type"] == "dynamic_point":
            raise NotImplementedError
        elif config["type"] == "static":
            point_cloud_range = np.array(config["point_cloud_range"], dtype=np.float32)
        else:
            raise NotImplementedError

        data_dict["point_cloud_range"] = point_cloud_range.astype(np.float32)
        return data_dict

    def calc_voxel_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.calc_voxel_size, config=config)

        point_cloud_range = data_dict["point_cloud_range"]
        grid_size = np.array(config["grid_size"], dtype=np.int64)
        voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / grid_size
        data_dict["voxel_size"] = voxel_size.astype(np.float32)
        data_dict["grid_size"] = grid_size
        return data_dict

    def sample_ray(self, data_dict=None, config=None):
        if data_dict is None:
            self.collider = getattr(scene_colliders, config["collider"]["type"])(
                **config["collider"]
            )
            return partial(self.sample_ray, config=config)

        scene_bbox = data_dict["point_cloud_range"]
        (
            iray_depth,
            iray_o,
            iray_d,
            iray_near,
            iray_far,
            iray_p,
            iray_idx,
        ) = ([], [], [], [], [], [], [])
        for cidx in range(len(data_dict["depth"])):
            ray_depth = data_dict["ray_depth"][cidx]
            ray_o = data_dict["ray_o"][cidx]
            ray_d = data_dict["ray_d"][cidx]
            ray_p = data_dict["ray_p"][cidx]
            ray_near, ray_far, ray_mask = self.collider(ray_o, ray_d, scene_bbox)

            if self.mode == "train":
                ray_mask = ray_mask & (ray_depth > 1e-3)
                valid_idx = np.nonzero(ray_mask)[0]
                assert len(valid_idx) > 0, "No ray is valid for camera %d" % cidx
                sampled_idx = np.random.choice(
                    len(valid_idx),
                    config["ray_nsample"],
                    replace=False,
                )
                sampled_idx = valid_idx[sampled_idx]

                ray_depth, ray_o, ray_d, ray_p, ray_near, ray_far = (
                    ray_depth[sampled_idx],
                    ray_o[sampled_idx],
                    ray_d[sampled_idx],
                    ray_p[sampled_idx],
                    ray_near[sampled_idx],
                    ray_far[sampled_idx],
                )

            iray_depth.append(ray_depth)
            iray_o.append(ray_o)
            iray_d.append(ray_d)
            iray_p.append(ray_p)
            iray_near.append(ray_near)
            iray_far.append(ray_far)
            iray_idx.append(np.full_like(ray_depth, cidx))

        assert len(iray_depth) > 0, "No ray is valid"
        data_dict["ray_depth"] = np.concatenate(iray_depth, axis=0)
        data_dict["ray_o"] = np.concatenate(iray_o, axis=0)
        data_dict["ray_d"] = np.concatenate(iray_d, axis=0)
        data_dict["ray_p"] = np.concatenate(iray_p, axis=0)
        data_dict["ray_near"] = np.concatenate(iray_near, axis=0)
        data_dict["ray_far"] = np.concatenate(iray_far, axis=0)
        data_dict["ray_idx"] = np.concatenate(iray_idx, axis=0)

        return data_dict

    def collect(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.collect, config=config)

        collected_data_dict = {}
        for key in config["keys"][self.mode]:
            # gt_boxes and gt_names are not necessary for test
            if key not in data_dict.keys():
                continue

            if key in ["img", "ori_img", "semantic_img"]:
                # (H, W, 3) -> (3, H, W)
                data_dict[key] = [
                    np.ascontiguousarray(_img.transpose(2, 0, 1).astype(np.float32))
                    / 255.0
                    for _img in data_dict[key]
                ]
            collected_data_dict[key] = data_dict[key]
        return collected_data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        return data_dict
