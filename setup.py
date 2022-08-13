#!/usr/bin/env python
from setuptools import setup, find_packages
from pathlib import Path

with open("requirements.txt") as f:
    deps = f.read().split("\n")

# surface distances dependency
deps.append(f"surface-distance-based-measures @ {(Path.cwd() / 'surface-distance').as_uri()}#egg=surface-distance-based-measures")

with open("requirements-dev.txt") as f:
    dev_deps = f.read().split("\n")

extras = {
   'wandb': ['wandb>=0.12.21'],
   'dev': dev_deps,
}

setup(
    name="uda-medical",
    version="0.1.0",
    description="Unsupervised Domain Adaptation",
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    author="Henri Iser",
    author_email="iserhenri@gmail.com",
    url="https://github.com/iserh/uda/tree/main",
    packages=find_packages(include=["uda*", "uda_wandb*"]),
    extras_require=extras,
    install_requires=deps,
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ]
)
