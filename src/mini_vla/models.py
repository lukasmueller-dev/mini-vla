"""Behavior cloning model: CNN image encoder + MLP state encoder + MLP action head."""

import torch
import torch.nn as nn
from typing import Sequence


class CNNEncoder(nn.Module):
    """Small ConvNet that maps an (C, H, W) image to a flat embedding."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        embed_dim: int,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for out_c, k, s in zip(channels, kernel_sizes, strides):
            layers += [nn.Conv2d(c, out_c, k, stride=s), nn.ReLU(inplace=True)]
            c = out_c
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # Compute flat dim eagerly so _proj is registered before any state_dict ops.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *image_size)
            flat_dim = self.flatten(self.conv(dummy)).shape[1]
        self._proj = nn.Linear(flat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._proj(self.flatten(self.conv(x))))


def _mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BCPolicy(nn.Module):
    """
    Behavior Cloning policy.

    Forward pass:
        image  : (B, C, H, W) float in [0, 1]
        state  : (B, state_dim) float
    Returns:
        action : (B, action_dim) float
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # CNN kwargs
        image_channels: int = 3,
        image_size: tuple[int, int] = (128, 128),
        cnn_channels: Sequence[int] = (32, 64, 64),
        cnn_kernel_sizes: Sequence[int] = (8, 4, 3),
        cnn_strides: Sequence[int] = (4, 2, 1),
        image_embed_dim: int = 256,
        # State encoder kwargs
        state_hidden_dims: Sequence[int] = (128, 128),
        state_embed_dim: int = 64,
        # Action head kwargs
        action_hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.image_encoder = CNNEncoder(
            in_channels=image_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            embed_dim=image_embed_dim,
            image_size=image_size,
        )
        self.state_encoder = _mlp(state_dim, state_hidden_dims, state_embed_dim)
        fused_dim = image_embed_dim + state_embed_dim
        self.action_head = _mlp(fused_dim, action_hidden_dims, action_dim)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_encoder(image)
        state_emb = torch.relu(self.state_encoder(state))
        fused = torch.cat([img_emb, state_emb], dim=-1)
        return self.action_head(fused)
