import os
from collections import defaultdict
from io import StringIO
from multiprocessing import Pool

import numpy as np
import src.utils as U
import torch

scene_root = "data/scannet"
rgbd_root = "data/scannet/rgbd"
scan_root = "data/scannet/scans"


def save_meta(scene_name, split):
    meta_data = defaultdict(dict)
    frame_list = os.listdir(os.path.join(rgbd_root, scene_name, "color"))
    frame_list = list(frame_list)
    frame_list = [frame for frame in frame_list if frame.endswith(".jpg")]
    frame_list.sort(key=lambda x: int(x.split(".")[0]))

    intrinsic = np.loadtxt(
        os.path.join(rgbd_root, scene_name, "intrinsic", "intrinsic_depth.txt")
    )
    if intrinsic is None or np.isnan(intrinsic).any() or np.isinf(intrinsic).any():
        return
    meta_data = defaultdict(dict)
    meta_data["scene_name"] = scene_name
    meta_data["intrinsic"] = intrinsic
    meta_data["frames"] = defaultdict(dict)
    for frame_name in frame_list:
        pose = np.loadtxt(
            os.path.join(
                rgbd_root,
                scene_name,
                "pose",
                frame_name.replace(".jpg", ".txt"),
            )
        )
        if pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            continue
        pose = np.linalg.inv(pose)
        if pose is None or np.isnan(pose).any() or np.isinf(pose).any():
            continue
        meta_data["frames"][frame_name]["color_path"] = os.path.join(
            rgbd_root, scene_name, "color", frame_name
        )
        meta_data["frames"][frame_name]["depth_path"] = os.path.join(
            rgbd_root, scene_name, "depth", frame_name.replace(".jpg", ".png")
        )
        meta_data["frames"][frame_name]["extrinsic"] = pose

    torch.save(
        meta_data,
        os.path.join(
            scene_root,
            "metadata",
            split,
            f"{scene_name}.pth",
        ),
    )


def main():
    for split in ["train", "val"]:
        scene_list = [
            filename.split(".")[0]
            for filename in os.listdir(os.path.join(scene_root, split))
        ]

        with Pool(processes=8) as pool:
            for scene_name in scene_list:
                pool.apply_async(save_meta, args=(scene_name, split))

            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
