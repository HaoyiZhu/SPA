import glob
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from io import StringIO

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

import spa.utils as U
from spa.data.components.processor import DataProcessor, augmentor_utils


class ScanNetMultiViewSPAPretrain(Dataset):
    def __init__(
        self,
        split="train",
        scene_root="data/scannet",
        frame_interval=10,
        num_cameras=5,
        loop=1,
        downsample_ratio=1,
        data_processor_cfg=None,
        batch_max_num_img=16,
        max_refetch=10,
        semantic_size=None,
        mode="train",
        scene_box_threshold=0.2,
        depth_area_threshold=0.1,
        **kwargs,
    ):
        super().__init__()
        self.scene_root = scene_root
        self.split = split
        self.loop = loop

        self.frame_interval = frame_interval
        self.num_cameras = num_cameras
        self.scene_interval = max(1, round(1 / downsample_ratio + 1e-6))
        self.semantic_size = semantic_size

        self.logger = U.RankedLogger(__name__, rank_zero_only=True)
        self.data_list = self.get_data_list()
        self.logger.info(
            "Totally {} x {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, downsample_ratio, split
            )
        )

        self.data_processor = DataProcessor(
            data_processor_cfg,
            mode=mode,
            logger=self.logger,
        )
        self.batch_max_num_img = batch_max_num_img
        self.max_refetch = max_refetch
        self.scene_box_threshold = scene_box_threshold
        self.depth_area_threshold = depth_area_threshold

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.scene_root, "metadata", self.split, "*.pth")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.scene_root, "metadata", split, "*.pth")
                )
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        metadata = torch.load(self.data_list[idx % len(self.data_list)])
        scene_name = metadata["scene_name"]
        intrinsic = metadata["intrinsic"]
        frames = metadata["frames"]

        num_cameras = np.random.randint(1, self.num_cameras + 1)

        frame_idx_start = np.random.randint(
            0, max(len(frames) - num_cameras * self.frame_interval + 1, 1)
        )
        frame_keys = list(frames.keys())[
            frame_idx_start : frame_idx_start
            + num_cameras * self.frame_interval : self.frame_interval
        ]
        frames = [frames[frame_key] for frame_key in frame_keys]

        intrinsics = np.stack([intrinsic for _ in range(len(frames))], axis=0)
        extrinsics = np.stack([frame["extrinsic"] for frame in frames], axis=0)

        assert (not np.isnan(extrinsics).any()) and (
            not np.isinf(extrinsics).any()
        ), "invalid extrinsics"

        data_dict = dict()

        depth = [
            U.io_utils.load_image(frame["depth_path"]).astype(np.float32) / 1000.0
            for frame in frames
        ]
        ori_rgb = [U.io_utils.load_image(frame["color_path"]) for frame in frames]
        h, w = depth[0].shape[-2:]
        rgb = [
            augmentor_utils.resize(
                _rgb,
                (w, h),
                "lanczos",
                "pillow",
            )
            for _rgb in ori_rgb
        ]
        if self.semantic_size is not None:
            semantic_rgb = [
                augmentor_utils.resize(
                    _rgb,
                    self.semantic_size,
                    "lanczos",
                    "pillow",
                )
                for _rgb in ori_rgb
            ]
            data_dict["semantic_img"] = semantic_rgb

        data_dict.update(
            dict(
                img=rgb,  # n, h, w, c
                ori_shape=np.stack([x.shape[:2] for x in rgb], axis=0),
                world2cam=extrinsics,
                cam2img=intrinsics,  # n, 4, 4
                depth=depth,  # n, h, w
            )
        )
        data_dict["trans3d_matrix"] = np.eye(4)
        data_dict["trans2d_matrix"] = [np.eye(4) for _img in data_dict["img"]]
        data_dict = self.data_processor.forward(data_dict=data_dict)
        for d in data_dict["depth"]:
            assert (d > 1e-3).astype(
                float
            ).mean() > self.depth_area_threshold, (
                f"valid depth area is small: {(d > 1e-3).astype(float).mean()}"
            )

        data_dict["scene_name"] = scene_name
        data_dict["frame_list"] = frame_keys
        data_dict["dataset_name"] = "scannet"

        if "point_cloud_range" in data_dict.keys():
            scene_box = data_dict["point_cloud_range"]
            assert (
                scene_box[3:] - scene_box[:3] > self.scene_box_threshold
            ).all(), f"too small scene box: {scene_box[3:] - scene_box[:3]}, scene: {scene_name}, frame: {frame_keys}"

        for key in data_dict:
            if isinstance(data_dict[key], list):
                data_dict[key] = np.stack(data_dict[key])
            if key in [
                "scene_name",
                "dataset_name",
                "frame_list",
                "point_cloud_range",
                "voxel_size",
                "grid_size",
                "ray_scale",
            ]:
                continue
            data_dict[key] = torch.from_numpy(data_dict[key]).float()

        return data_dict

    def _collate_fn(self, batch, trunc_batch=True):
        if not isinstance(batch, Sequence):
            raise TypeError(f"{batch.dtype} is not supported.")

        if trunc_batch and self.batch_max_num_img > 0:
            accum_num_imgs = 0
            ret_batches = []
            for batch_id, data in enumerate(batch):
                num_imgs = len(data["img"])
                if accum_num_imgs + num_imgs > self.batch_max_num_img:
                    # log.info(
                    #     f"Truncating batch {batch_id} since accum_num_imgs {accum_num_imgs} + num_imgs {num_imgs} > batch_max_num_img {self.batch_max_num_img}."
                    # )
                    continue
                accum_num_imgs += num_imgs
                ret_batches.append(data)
            return self._collate_fn(ret_batches, trunc_batch=False)

        return_dict = dict()
        return_dict["batch_size"] = len(batch)

        for k in batch[0]:
            return_dict[k] = [d[k] for d in batch]

        return return_dict

    def get_data_name(self, scene_path):
        return os.path.basename(scene_path).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        try:
            data_dict = self.get_data(idx)
        except Exception as e:
            return None, e
        return data_dict, None

    def __getitem__(self, idx):
        for _ in range(self.max_refetch):
            new_idx = idx * self.scene_interval + np.random.randint(
                0, self.scene_interval
            )
            data, e = self.prepare_train_data(new_idx)
            if data is None:
                self.logger.warning(
                    f"Failed to load data from {self.data_list[new_idx % len(self.data_list)]} for error {e}."
                )
                idx = self._rand_another()
                continue
            else:
                return data
        raise e

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def __len__(self):
        return int(len(self.data_list) * self.loop) // self.scene_interval
