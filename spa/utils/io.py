import json
import os
import pickle
from io import BytesIO

import cv2
import imageio.v3 as iio
import numpy as np
import torch


def load_bytes(filename):
    return open(filename, "rb").read()


def load_mp4(filename):
    return iio.imread(filename, extension=".mp4")


def load_text(filename):
    return open(filename, "r").read().splitlines()


def load_numpy_text(filename):
    return np.loadtxt(filename)


def listdir(dir):
    return os.listdir(dir)


def load_json(filename):
    return json.load(open(filename))


def dump_json(filename, data, backend="local"):
    json.dump(data, open(filename, "w"))


def load_numpy_pickle(filename):
    array = np.load(filename, allow_pickle=True)

    try:
        return array.item()
    except:
        return array


def load_numpy(filename):
    return np.load(filename)


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def load_image(filename):
    if ".jpg" in filename or ".JPG" in filename:
        image = iio.imread(filename)
    elif ".png" in filename or ".PNG" in filename:  # for depth images
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return image


def exists(path):
    return os.path.exists(path)


def isdir(path):
    return os.path.isdir(path)


def imwrite(img, path):
    iio.imwrite(path, img)


def load_pth(
    path,
    map_location="cpu",
):
    return torch.load(path, map_location=map_location)
