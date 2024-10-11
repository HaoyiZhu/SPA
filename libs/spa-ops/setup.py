import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists(".git"):
        return "0000000"

    cmd_out = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode("utf-8")[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=[os.path.join(*module.split("."), src) for src in sources],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                "-O2",
                "-std=c++17",
            ],
        },
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, "w") as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == "__main__":
    version = "0.6.0+%s" % get_git_commit_number()
    setup(
        name="spa_ops",
        version=version,
        description="SPA ops",
        install_requires=[
            "numpy",
            "llvmlite",
            "numba",
            "tensorboardX",
            "easydict",
            "pyyaml",
            "scikit-image",
            "tqdm",
        ],
        author="Haoyi Zhu",
        author_email="hyizhu1108@gmail.com",
        license="Apache License 2.0",
        packages=find_packages(),
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name="voxel_pool_ext",
                module="voxel_pool",
                sources=[
                    "src/voxel_pool.cpp",
                    "src/voxel_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="grid_sampler_cuda",
                module="grid_sampler",
                sources=[
                    "src/grid_sampler.cpp",
                    "src/grid_sampler_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="ms_deform_attn_cuda",
                module="deform_attn",
                sources=[
                    "src/ms_deform_attn.cpp",
                    "src/ms_deform_attn_cuda.cu",
                ],
            ),
        ],
    )
