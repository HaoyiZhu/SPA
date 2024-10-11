import copy
import os
from collections.abc import Mapping, Sequence
from typing import Union

import numpy as np
import torch


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, str):
        # note that str is also a kind of sequence, judgement should before sequence
        return data
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
        return torch.from_numpy(data)
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
        return torch.from_numpy(data).long()
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
        return torch.from_numpy(data).float()
    elif isinstance(data, Mapping):
        result = {sub_key: to_tensor(item) for sub_key, item in data.items()}
        return result
    elif isinstance(data, Sequence):
        result = [to_tensor(item) for item in data]
        return result
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


class Collect:
    """Collect data from the loader relevant to the specific task.
    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.
    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:
            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "label",
            "original_shape",
            "img_shape",
            "pad_shape",
            "flip_direction",
            "img_norm_cfg",
        ),
        meta_name="img_metas",
        nested=False,
    ):
        self.keys = keys
        self.meta_keys = meta_keys if meta_keys is not None else []
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]
        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = meta
        if self.nested:
            for k in data:
                data[k] = [data[k]]
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"keys={self.keys}, meta_keys={self.meta_keys}, "
            f"nested={self.nested})"
        )


class Copy:
    def __init__(self, keys_dict=dict()):
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.keys is None:
            return to_tensor(results)

        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


class NormalizeColor:
    def __init__(self, keys=["obs_imgs"], mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)

    def __call__(self, data_dict):
        for key in self.keys:
            assert (
                data_dict[key].shape[-3] == 3 or data_dict[key].shape[-3] == 4
            ), f"Only support RGB or RGB-D, but key {key} has shape {data_dict[key].shape}"
            data_dict[key][..., :3, :, :] = data_dict[key][..., :3, :, :] / 255.0
            data_dict[key][..., : self.mean.shape[1], :, :] = (
                data_dict[key][..., : self.mean.shape[1], :, :] - self.mean
            ) / self.std

        return data_dict


class Compose:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
            if data_dict is None:
                return None
        return data_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
