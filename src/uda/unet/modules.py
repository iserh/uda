from typing import Union

import torch.nn as nn


def ConvNd(dim: int, *args, **kwargs) -> Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"Conv{dim}d")
    return cls(*args, **kwargs)


def ConvTransposeNd(dim: int, *args, **kwargs) -> Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"ConvTranspose{dim}d")
    return cls(*args, **kwargs)


def BatchNormNd(dim: int, *args, **kwargs) -> Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"BatchNorm{dim}d")
    return cls(*args, **kwargs)


def MaxPoolNd(dim: int, *args, **kwargs) -> Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"MaxPool{dim}d")
    return cls(*args, **kwargs)
