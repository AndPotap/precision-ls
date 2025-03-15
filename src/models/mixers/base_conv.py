# Adapted from https://github.com/HazyResearch/zoology

from typing import List, Union

import torch
import torch.nn as nn

from .convs import (
    ShortConvolution,
    LongConvolution,
    ImplicitLongConvolution,
)


class BaseConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_size: Union[int, List[int]] = -1,
        layer_idx: int = None,
        conv_type: str = "short",  # short, long, implicit
        resid: bool = True,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx = layer_idx
        self.resid = resid
        self.conv_type = conv_type
        self.causal = causal

        self.projection = nn.Linear(self.d_model, self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # (Main, resid)
        # init = (1, 0)
        if self.resid:
            self.branch_weights = nn.Parameter(torch.zeros(2), requires_grad=True)
            self.branch_weights.data[0] = 1

        # support for different kernel sizes per layer
        if isinstance(kernel_size, List):
            if layer_idx is None or layer_idx >= len(kernel_size):
                raise ValueError(
                    "kernel_size must be an int or a list of ints with length equal to the number of layers"
                )
            kernel_size = kernel_size[layer_idx]

        # prepare convolution
        if kernel_size == -1:
            if conv_type == "implicit":
                conv = ImplicitLongConvolution
            elif conv_type == "long":
                conv = LongConvolution
            elif conv_type == "short":
                conv = ShortConvolution
            print(f"Conv type = {conv_type}")
            self.conv = conv(d_model, l_max=l_max, resid=self.resid, causal=self.causal)
        else:
            self.conv = ShortConvolution(
                d_model, kernel_size=kernel_size, resid=self.resid, causal=self.causal
            )

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u_conv = self.conv(self.in_proj(u))
        u_proj = self.projection(u)
        y = self.out_proj(u_conv * u_proj)
        if self.resid:
            return self.branch_weights[0] * y + self.branch_weights[1] * u
        else:
            return y


class BaseConvLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.base_conv = BaseConv(
            d_model=config.n_embd,
            l_max=config.block_size,
            kernel_size=-1,
            conv_type=config.conv_type,
            resid=config.use_resid,
            causal=config.causal,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.base_conv(x)
        x = self.dropout(x)
        return x
