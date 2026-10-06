"""Optional C++ kernels: install with the scanner's Python via pip install ./native."""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="kinect-scanner-native",
    version="0.2.0",
    description="Fused calibrated RGB-D preparation and weighted TSDF kernels",
    python_requires=">=3.10",
    ext_modules=[
        Pybind11Extension(
            "_kinect_native",
            ["kernels.cpp"],
            depends=["weighted_fusion.h"],
            cxx_std=17,
            # Preserve the reference's rounding around projection boundaries.
            # Fast math/FMA contraction can change an OpenCV remap bin.
            extra_compile_args=["/O2", "/fp:strict"]
            if sys.platform == "win32"
            else ["-O3", "-ffp-contract=off"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    packages=[],
)
