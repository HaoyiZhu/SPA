import argparse
import os

import imageio.v3 as iio
import numpy as np
import torch
from einops import rearrange
from skimage.transform import resize

from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
from spa.models.components.spa import get_pca_map


def main():
    parser = argparse.ArgumentParser(description="Visualize feature maps.")
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=448,
        help="Resolution to resize images to before processing. Defaults to 448.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="visualize",
        help="Path to save the feature map visualization results.",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = spa_vit_large_patch16(img_size=args.img_size, pretrained=True)
    model.eval()
    model.freeze()
    model = model.to(device)

    images = [
        iio.imread(os.path.join(args.image_folder, f))[..., :3]
        for f in sorted(os.listdir(args.image_folder))
    ]
    images = np.stack(
        [
            (resize(image, (448, 448), anti_aliasing=True) * 255).astype(np.uint8)
            for image in images
        ]
    )
    # images = np.stack(images) # / 255.
    images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device) / 255.0
    images = torch.nn.functional.interpolate(images, size=(448, 448), mode="bilinear")
    # n c h w
    feature_map = model(images, feature_map=True, cat_cls=False)
    feature_map = torch.nn.functional.interpolate(
        feature_map, size=(224, 224), mode="bilinear"
    )
    feature_map = rearrange(feature_map, "n c h w -> 1 (n h) w c")

    feature_pca = get_pca_map(feature_map)  # (n h) w c
    feature_pca = (feature_pca * 255).astype(np.uint8).squeeze()

    os.makedirs(args.output_folder, exist_ok=True)
    iio.imwrite(
        os.path.join(
            args.output_folder,
            f"{os.path.basename(args.image_folder)}_feature_map_vis.png",
        ),
        feature_pca,
    )


if __name__ == "__main__":
    main()
