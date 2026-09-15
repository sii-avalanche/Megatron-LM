# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        if config.layernorm_zero_centered_gamma:
            return _ZeroCenteredNorm(hidden_size, eps, config.normalization)

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)


class _ZeroCenteredNorm(torch.nn.Module):
    """Torch norm equivalent of Megatron's zero-centered gamma convention."""

    def __init__(self, hidden_size, eps, normalization):
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps
        self.normalization = normalization
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.bias = (
            torch.nn.Parameter(torch.zeros(hidden_size)) if normalization == "LayerNorm" else None
        )

    def forward(self, x):
        # Megatron stores gamma around zero and adds one at use time.
        weight = self.weight + 1
        if self.normalization == "LayerNorm":
            return torch.nn.functional.layer_norm(
                x, self.normalized_shape, weight, self.bias, self.eps
            )
        return torch.nn.functional.rms_norm(x, self.normalized_shape, weight, self.eps)


class L2Norm(torch.nn.Module):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)
