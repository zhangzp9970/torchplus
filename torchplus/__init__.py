from torchplus import datasets
from torchplus import nn
from torchplus import utils
from .version import __version__ as torch_plus_version
try:
    from torch import __version__ as torch_version
except ImportError:
    raise RuntimeError('Pytorch not found !')
try:
    from torchvision import __version__ as torchvision_version
except ImportError:
    raise RuntimeError('torchvision not found !')

# check torch version and torchvision version
if torch_version < '1.8.1':
    raise RuntimeWarning('Pytorch version should be greater than 1.8.1 !')
if torchvision_version < '0.9.1':
    raise RuntimeWarning('torchvision version should be greater than 0.9.1 !')
