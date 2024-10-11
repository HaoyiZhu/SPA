#!/usr/bin/env python
import os
import pathlib

import pkg_resources
from setuptools import find_packages, setup

PKG_NAME = "spa"
VERSION = "0.1"
EXTRAS = {}


def _read_file(fname):
    # this_dir = os.path.abspath(os.path.dirname(__file__))
    # with open(os.path.join(this_dir, fname)) as f:
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


setup(
    name=PKG_NAME,
    version=VERSION,
    author="Haoyi Zhu",
    author_email="hyizhu1108@gmail.com",
    url="https://github.com/HaoyiZhu/SPA",
    description="SPA: 3D Spatial-Awareness Enables Effective Embodied Representation",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Embodied AI",
        "Representation Learning",
    ],
    license="MIT License",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "train_command = spa.train:main",
        ]
    },
)
