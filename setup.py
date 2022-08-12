#!/usr/bin/env python
from setuptools import setup

with open("requirements.txt") as f:
    deps = f.read().split("\n")

# surface-distance
deps.append("surface-distance-based-measures")
dep_links = ["git+ssh://git@github.com/deepmind/surface-distance#egg=surface-distance-based-measures"]

with open("dev-requirements.txt") as f:
    dev_deps = f.read().split("\n")

extras = {
   'wandb': ['wandb>=0.12.21'],
   'dev': dev_deps,
}

setup(
    name="uda",
    version="0.1.0",
    description="Unsupervised Domain Adaptation",
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    author="Henri Iser",
    author_email="iserhenri@gmail.com",
    url="https://github.com/ndoll1998/uda/tree/master",
    extras_require=extras,
    install_requires=deps,
    dependency_links=dep_links,
    packages=['uda'],
    package_dir={'uda': 'uda', 'uda_wandb': 'uda_wandb'},
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Analysis"
    ]
)
