from typing import Tuple

from torch import Tensor

__all__ = ['equivariant_1_to_0', 'equivariant_1_to_1', 'equivariant_1_to_2',
           'equivariant_2_to_1', 'equivariant_2_to_2', 'equivariant_2_to_2_symmetric']


def equivariant_1_to_0(x: Tensor) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 1D -> 0D

    :param x: Input tensor of shape (B × D × N)
    :return: Output of the 1 linear basis function, shape (B × D)
    """
    e1 = x.sum(dim=-1)
    return e1,


def equivariant_1_to_1(x: Tensor, normalize: bool = False) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 1D -> 1D

    :param x: Input tensor of shape (B × D × N)
    :param normalize: Normalize w.r.t. N
    :return: Output of the 2 linear basis functions, shape (B × D × N)
    """
    e1 = x
    e2 = x.sum(dim=-1, keepdim=True).expand_as(x)
    if normalize:
        n = x.shape[-1]
        return e1, e2/n
    else:
        return e1, e2


def equivariant_1_to_2(x: Tensor, normalize: bool = False) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 1D -> 2D

    :param x: Input tensor of shape (B × D × N)
    :param normalize: Normalize w.r.t. N
    :return: Output of the 5 linear basis functions, shape (B × D × N × N)
    """
    n = x.shape[-1]
    e1 = x.diag_embed()
    sum_all = x.sum(dim=-1, keepdim=True)
    e2 = sum_all.expand(-1, -1, n).diag_embed()
    e3 = x.unsqueeze(-1).expand(-1, -1, -1, n)
    e4 = e3.permute(0, 1, 3, 2)
    e5 = sum_all.unsqueeze(-1).expand(-1, -1, n, n)
    if normalize:
        return e1, e2/n, e3, e4, e5/n
    else:
        return e1, e2, e3, e4, e5


def equivariant_2_to_1(x: Tensor, normalize: bool = False) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 2D -> 1D

    :param x: Input tensor of shape (B × D × N × N)
    :param normalize: Normalize w.r.t. N
    :return: Output of the 5 linear basis functions, shape (B × D × N)
    """
    n = x.shape[-1]
    e1 = x.diagonal(dim1=-1, dim2=-2)
    e2 = e1.sum(dim=-1, keepdim=True).expand(-1, -1, n)
    e3 = x.sum(dim=-1)
    e4 = x.sum(dim=-2)
    e5 = e4.sum(dim=-1, keepdim=True).expand(-1, -1, n)
    if normalize:
        return e1, e2/n, e3/n, e4/n, e5/n**2
    else:
        return e1, e2, e3, e4, e5


def equivariant_2_to_2(x: Tensor, normalize: bool = False) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 2D -> 2D

    :param x: Input tensor of shape (B × D × N × N)
    :param normalize: Normalize w.r.t. N
    :return: Output of the 15 linear basis functions, shape (B × D × N × N)
    """
    n = x.shape[-1]
    e1 = x
    e2 = x.permute(0, 1, 3, 2)
    dia = x.diagonal(dim1=-2, dim2=-1)
    e3 = dia.diag_embed()
    row_sum = x.sum(dim=-1)
    e4 = row_sum.unsqueeze(-2).expand(-1, -1, n, -1)
    e5 = row_sum.unsqueeze(-1).expand(-1, -1, -1, n)
    e6 = row_sum.diag_embed()
    col_sum = x.sum(dim=-2)
    e7 = col_sum.unsqueeze(-2).expand(-1, -1, n, -1)
    e8 = col_sum.unsqueeze(-1).expand(-1, -1, -1, n)
    e9 = col_sum.diag_embed()
    all_sum = x.sum(dim=(-2, -1), keepdim=True)
    e10 = all_sum.expand(-1, -1, n, n)
    e11 = all_sum.squeeze(-1).expand(-1, -1, n).diag_embed()
    dia_sum = dia.sum(dim=-1, keepdim=True)
    e12 = dia_sum.unsqueeze(-1).expand(-1, -1, n, n)
    e13 = dia_sum.expand(-1, -1, n).diag_embed()
    e14 = dia.unsqueeze(-2).expand(-1, -1, n, -1)
    e15 = dia.unsqueeze(-1).expand(-1, -1, -1, n)
    if normalize:
        return e1, e2/n, e3/n, e4/n, e5/n**2, e6/n, e7/n, e8/n, e9/n, e10, e11, e12, e13, e14/n, e15/n**2
    else:
        return e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15


def equivariant_2_to_2_symmetric(x: Tensor, normalize: bool = False) -> Tuple[Tensor, ...]:
    """
    Equivariant linear basis for 2D -> 2D, symmetric matrices

    :param x: Input tensor of shape (B × D × N × N)
    :param normalize: Normalize w.r.t. N
    :return: Output of the 11 linear basis functions, shape (B × D × N × N)
    """
    n = x.shape[-1]
    e1 = x
    dia = x.diagonal(dim1=-2, dim2=-1)
    e3 = dia.diag_embed()
    row_sum = x.sum(dim=-1)
    e4 = row_sum.unsqueeze(-2).expand(-1, -1, n, -1)
    e5 = row_sum.unsqueeze(-1).expand(-1, -1, -1, n)
    e6 = row_sum.diag_embed()
    all_sum = x.sum(dim=(-2, -1), keepdim=True)
    e10 = all_sum.expand(-1, -1, n, n)
    e11 = all_sum.squeeze(-1).expand(-1, -1, n).diag_embed()
    dia_sum = dia.sum(dim=-1, keepdim=True)
    e12 = dia_sum.unsqueeze(-1).expand(-1, -1, n, n)
    e13 = dia_sum.expand(-1, -1, n).diag_embed()
    e14 = dia.unsqueeze(-2).expand(-1, -1, n, -1)
    e15 = dia.unsqueeze(-1).expand(-1, -1, -1, n)
    if normalize:
        return e1, e3/n, e4/n, e5/n**2, e6/n, e10, e11, e12, e13, e14/n, e15/n**2
    else:
        return e1, e3, e4, e5, e6, e10, e11, e12, e13, e14, e15
