import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, init

from .functional import *

__all__ = ['LinearEquivariant1to0', 'LinearEquivariant1to1', 'LinearEquivariant1to2',
           'LinearEquivariant2to1', 'LinearEquivariant2to2', 'LinearEquivariant2to2Symmetric']


class LinearEquivariant1to0(Module):
    """Equivariant linear layer 1D -> 0D

    Input shape :code:`(batch, in_features, dim)`, output shape :code:`(batch, out_features)`.

    :param in_features: Input features
    :param out_features: Output features
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(LinearEquivariant1to0, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x, = equivariant_1_to_0(x)
        return F.linear(x, self.weight, self.bias)

    @property
    def basis_elements(self) -> int:
        return 1

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LinearEquivariant1to1(Module):
    """Equivariant linear layer 1D -> 1D

    Input shape :code:`(batch, in_features, dim)`, output shape :code:`(batch, out_features, dim)`.

    :param in_features: Input features
    :param out_features: Output features
    :param normalize: Whether to normalize basis
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = False, device=None, dtype=None):
        super(LinearEquivariant1to1, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features * self.basis_elements, 1), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(equivariant_1_to_1(x, self.normalize), dim=1)
        return F.conv1d(x, self.weight, self.bias)

    @property
    def basis_elements(self) -> int:
        return 2

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] // self.basis_elements

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LinearEquivariant1to2(Module):
    """Equivariant linear layer 1D -> 2D

    Input shape :code:`(batch, in_features, dim)`, output shape :code:`(batch, out_features, dim, dim)`.

    :param in_features: Input features
    :param out_features: Output features
    :param normalize: Whether to normalize basis
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = False, device=None, dtype=None):
        super(LinearEquivariant1to2, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features * self.basis_elements, 1, 1), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(equivariant_1_to_2(x, self.normalize), dim=1)
        return F.conv2d(x, self.weight, self.bias)

    @property
    def basis_elements(self) -> int:
        return 5

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] // self.basis_elements

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LinearEquivariant2to1(Module):
    """Equivariant linear layer 2D -> 1D

    Input shape :code:`(batch, in_features, dim, dim)`, output shape :code:`(batch, out_features, dim)`.

    :param in_features: Input features
    :param out_features: Output features
    :param normalize: Whether to normalize basis
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = False, device=None, dtype=None):
        super(LinearEquivariant2to1, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features * self.basis_elements, 1), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(equivariant_2_to_1(x, self.normalize), dim=1)
        return F.conv1d(x, self.weight, self.bias)

    @property
    def basis_elements(self) -> int:
        return 5

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] // self.basis_elements

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LinearEquivariant2to2(Module):
    """Equivariant linear layer 2D -> 2D

    Input shape :code:`(batch, in_features, dim, dim)`, output shape :code:`(batch, out_features, dim, dim)`.

    :param in_features: Input features
    :param out_features: Output features
    :param normalize: Whether to normalize basis
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = False, device=None, dtype=None):
        super(LinearEquivariant2to2, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features * self.basis_elements, 1, 1), **factory_kwargs))
        self.bias_all = Parameter(torch.empty(out_features, **factory_kwargs))
        self.bias_diag = Parameter(torch.empty((out_features, 1), **factory_kwargs))
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias_all, -bound, bound)
        init.uniform_(self.bias_diag, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(equivariant_2_to_2(x, self.normalize), dim=1)
        diag_bias = self.bias_diag.expand(-1, x.shape[-1]).diag_embed()
        return F.conv2d(x, self.weight, self.bias_all) + diag_bias

    @property
    def basis_elements(self) -> int:
        return 15

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] // self.basis_elements

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LinearEquivariant2to2Symmetric(Module):
    """Equivariant linear layer 2D -> 2D, symmetric matrices

    Input shape :code:`(batch, in_features, dim, dim)`, output shape :code:`(batch, out_features, dim, dim)`.

    :param in_features: Input features
    :param out_features: Output features
    :param normalize: Whether to normalize basis
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = False, device=None, dtype=None):
        super(LinearEquivariant2to2Symmetric, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features * self.basis_elements, 1, 1), **factory_kwargs))
        self.bias_all = Parameter(torch.empty(out_features, **factory_kwargs))
        self.bias_diag = Parameter(torch.empty((out_features, 1), **factory_kwargs))
        self.normalize = normalize
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.bias_all, -bound, bound)
        init.uniform_(self.bias_diag, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(equivariant_2_to_2_symmetric(x, self.normalize), dim=1)
        diag_bias = self.bias_diag.expand(-1, x.shape[-1]).diag_embed()
        return F.conv2d(x, self.weight, self.bias_all) + diag_bias

    @property
    def basis_elements(self) -> int:
        return 11

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] // self.basis_elements

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
