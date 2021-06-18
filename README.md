# Pytorch *Equivariant*

Implementation of linear equivariant network layers for Pytorch.

## Example

Usage is similar to Pytorch's `Linear` layer.

```python
import torch
from equivariant.nn import LinearEquivariant2to2

x = torch.rand(32, 4, 12, 12)  # shape (batch, features, dim, dim)
equi = LinearEquivariant2to2(in_features=4, out_features=3)
y = equi(x)  # shape (32, 3, 12, 12)
```

## Reference

Haggai Maron, Heli Ben-Hamu, Nadav Shamir, Yaron Lipman. **Invariant and Equivariant Graph Networks.** *International Conference on Learning Representations (ICLR) 2019.* http://arxiv.org/abs/1812.09902
