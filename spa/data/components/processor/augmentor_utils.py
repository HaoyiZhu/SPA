import cv2
import numpy as np
import torch
from PIL import Image


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def angle2matrix(angle):
    """
    Args:
        angle: angle along z-axis, angle increases x ==> y
    Returns:
        rot_matrix: (3x3 Tensor) rotation matrix
    """
    angle, is_numpy = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.tensor([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    return rot_matrix.numpy() if is_numpy else rot_matrix


def global_drop(points, drop_ratio, prob=0.5):
    enable = np.random.choice(
        [False, True],
        replace=False,
        p=[1 - prob, prob],
    )
    if enable:
        choice = np.arange(0, len(points), dtype=np.int32)
        choice = np.random.choice(
            choice, int((1 - drop_ratio) * len(points)), replace=False
        )
        points = points[choice]
    return points


def random_flip_along_x(gt_boxes, points, prob=0.5):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[1 - prob, prob])
    matrix = np.eye(4)
    if enable:
        if gt_boxes is not None:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 8] = -gt_boxes[:, 8]
        if points is not None:
            points[:, 1] = -points[:, 1]
        matrix[1, 1] = -1

    return gt_boxes, points, matrix


def random_flip_along_y(gt_boxes, points, prob=0.5):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[1 - prob, prob])
    matrix = np.eye(4)
    if enable:
        if gt_boxes is not None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7] = -gt_boxes[:, 7]
        if points is not None:
            points[:, 0] = -points[:, 0]
        matrix[0, 0] = -1

    return gt_boxes, points, matrix


def global_rotation(gt_boxes, points, rot_range, prob=0.5):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    enable = np.random.choice(
        [False, True],
        replace=False,
        p=[1 - prob, prob],
    )
    matrix = np.eye(4)
    if enable:
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        if points is not None:
            points = rotate_points_along_z(
                points[np.newaxis, :, :], np.array([noise_rotation])
            )[0]
        if gt_boxes is not None:
            gt_boxes[:, 0:3] = rotate_points_along_z(
                gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation])
            )[0]
            gt_boxes[:, 6] += noise_rotation
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7:9] = rotate_points_along_z(
                    np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
                        np.newaxis, :, :
                    ],
                    np.array([noise_rotation]),
                )[0][:, 0:2]
        matrix[:3, :3] = angle2matrix(np.array(noise_rotation))

    return gt_boxes, points, matrix


def global_scaling(gt_boxes, points, scale_range, prob=0.5):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    enable = np.random.choice(
        [False, True],
        replace=False,
        p=[1 - prob, prob],
    )
    matrix = np.eye(4)
    if enable:
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        if points is not None:
            points[:, :3] *= noise_scale
        if gt_boxes is not None:
            gt_boxes[:, :6] *= noise_scale
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7:] *= noise_scale
        matrix[:3, :3] *= noise_scale

    return gt_boxes, points, matrix


def global_translation(gt_boxes, points, translate_std, prob=0.5):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    enable = np.random.choice(
        [False, True],
        replace=False,
        p=[1 - prob, prob],
    )
    matrix = np.eye(4)
    if enable:
        noise_translate = np.array(
            [
                np.random.normal(0, translate_std[0], 1),
                np.random.normal(0, translate_std[1], 1),
                np.random.normal(0, translate_std[2], 1),
            ],
            dtype=np.float32,
        ).T
        if points is not None:
            points[:, :3] += noise_translate
        if gt_boxes is not None:
            gt_boxes[:, :3] += noise_translate
        matrix[:3, 3] = noise_translate.squeeze()

    return gt_boxes, points, matrix


def create_grid_mask(
    size,
    mask_ratio=0.5,
    probability=0.7,
    rotate_angle=0,
    add_noise=False,
    rise_with_epoch=False,
    epoch_state=None,
):
    if rise_with_epoch:
        assert epoch_state is not None
        probability = epoch_state[0] / epoch_state[1] * probability

    ratio = 1 - mask_ratio
    h, w = size[:2]
    hh, ww = int(1.5 * h), int(1.5 * w)
    d = np.random.randint(2, h)
    l = min(max(int(d * ratio + 0.5), 1), d - 1)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)

    mask = np.ones((hh, ww), dtype=np.uint8)
    for i in range(hh // d):
        s = d * i + st_h
        t = min(s + l, hh)
        mask[s:t, :] = 0

    for i in range(ww // d):
        s = d * i + st_w
        t = min(s + l, ww)
        mask[:, s:t] = 0

    if rotate_angle > 0:
        angle = np.random.randint(rotate_angle + 1)
        c_x, c_y = (ww - 1) * 0.5, (hh - 1) * 0.5
        matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1)
        mask = cv2.warpAffine(mask, matrix, (ww, hh))

    mask = mask[(hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w]
    mask = 1 - mask  # 1 for visible, 0 for mask

    if add_noise:
        # add noise for mask region
        noise = 2 * (np.random.rand(*mask.shape) - 0.5)  # [-1, 1]
        mask_noise = noise * (1 - mask)
    else:
        mask_noise = np.zeros_like(mask)

    return mask, mask_noise


def crop(img, start_h, start_w, crop_h, crop_w):
    img_src = np.zeros((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype)
    hsize, wsize = crop_h, crop_w
    dh, dw, sh, sw = start_h, start_w, 0, 0
    if dh < 0:
        sh = -dh
        hsize += dh
        dh = 0
    if dh + hsize > img.shape[0]:
        hsize = img.shape[0] - dh
    if dw < 0:
        sw = -dw
        wsize += dw
        dw = 0
    if dw + wsize > img.shape[1]:
        wsize = img.shape[1] - dw
    img_src[sh : sh + hsize, sw : sw + wsize] = img[dh : dh + hsize, dw : dw + wsize]
    return img_src


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

pillow_interp_codes = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
}


def resize(img, size, interpolation="bilinear", backend="cv2"):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    if h == size[1] and w == size[0]:
        return img

    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, interpolation=cv2_interp_codes[interpolation]
        )

    return resized_img


class ColorJitter(object):
    def __init__(
        self,
        contrast=[0.5, 1.5],
        saturation=[0.5, 1.5],
        hue=[-0.05, 0.05],
        brightness=[0.875, 1.125],
        p=0.5,
    ):
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.brightness = brightness
        self.p = p
        self.reset_params()

    def rgb_to_grayscale(self, img):
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
        return l_img

    def adjust_brightness(self, img, brightness_factor):
        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def adjust_contrast(self, img, contrast_factor):
        mean = torch.mean(self.rgb_to_grayscale(img), dim=(-3, -2, -1), keepdim=True)
        return self._blend(img, mean, contrast_factor)

    def adjust_hue(self, img, hue_factor):
        img = self._rgb2hsv(img)
        h, s, v = img.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        img = torch.stack((h, s, v), dim=-3)
        img_hue_adj = self._hsv2rgb(img)
        return img_hue_adj

    def adjust_saturation(self, img, saturation_factor):
        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)

    def _blend(self, img1, img2, ratio):
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, 1).to(img1.dtype)

    def _rgb2hsv(self, img):
        r, g, b = img.unbind(dim=-3)

        # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
        # src/libImaging/Convert.c#L330
        maxc = torch.max(img, dim=-3).values
        minc = torch.min(img, dim=-3).values

        # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
        # from happening in the results, because
        #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
        #   + H channel has division by `(maxc - minc)`.
        #
        # Instead of overwriting NaN afterwards, we just prevent it from occurring, so
        # we don't need to deal with it in case we save the NaN in a buffer in
        # backprop, if it is ever supported, but it doesn't hurt to do so.
        eqc = maxc == minc

        cr = maxc - minc
        # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
        ones = torch.ones_like(maxc)
        s = cr / torch.where(eqc, ones, maxc)
        # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
        # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
        # would not matter what values `rc`, `gc`, and `bc` have here, and thus
        # replacing denominator with 1 when `eqc` is fine.
        cr_divisor = torch.where(eqc, ones, cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = torch.fmod((h / 6.0 + 1.0), 1.0)
        return torch.stack((h, s, maxc), dim=-3)

    def _hsv2rgb(self, img):
        h, s, v = img.unbind(dim=-3)
        i = torch.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.to(dtype=torch.int32)

        p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
        q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
        t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6

        mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

        a1 = torch.stack((v, q, p, p, t, v), dim=-3)
        a2 = torch.stack((t, v, v, q, p, p), dim=-3)
        a3 = torch.stack((p, p, t, v, v, q), dim=-3)
        a4 = torch.stack((a1, a2, a3), dim=-4)

        return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)

    def reset_params(self):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        r = torch.rand(7)
        b = (
            float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
            if r[0] < self.p
            else None
        )
        c = (
            float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
            if r[2] < self.p or r[5] < self.p
            else None
        )
        s = (
            float(torch.empty(1).uniform_(self.saturation[0], self.saturation[1]))
            if r[3] < self.p
            else None
        )
        h = (
            float(torch.empty(1).uniform_(self.hue[0], self.hue[1]))
            if r[4] < self.p
            else None
        )
        p = torch.randperm(3) if r[6] < self.p else None
        self.jitter_params = (r, b, c, s, h, p)

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        r, b, c, s, h, p = self.jitter_params

        if r[0] < self.p:
            img = self.adjust_brightness(img, b)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                img = self.adjust_contrast(img, c)

        if r[3] < self.p:
            img = self.adjust_saturation(img, s)

        if r[4] < self.p:
            img = self.adjust_hue(img, h)

        if not contrast_before:
            if r[5] < self.p:
                img = self.adjust_contrast(img, c)

        if r[6] < self.p:
            img = img[..., p, :, :]

        return img
