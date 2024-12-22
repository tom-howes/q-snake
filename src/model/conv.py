import torch

from torch import nn


class Block(nn.Module):
    def __init__(self, d_model, hidden_size=None, norm_eps=1e-6):
        super(Block, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + self.ffn(x)


class DQN(nn.Module):
    def __init__(self, dims=(64, 128, 256), depths=(1, 2, 2)):
        super(DQN, self).__init__()
        layers = [nn.Conv2d(3, dims[0], kernel_size=7, padding=3)]

        for i in range(len(dims) - 1):
            layers += [
                Block(dims[i]) for _ in range(depths[i])
            ]
            layers += [
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            ]

        layers += [
            Block(dims[-1]) for _ in range(depths[-1])
        ]
        self.conv = nn.Sequential(*layers)

        self.ffn = nn.Sequential(
            nn.Linear(dims[-1], 4 * dims[-1]),
            nn.GELU(),
            nn.Linear(4 * dims[-1], 3)
        )

    def forward(self, x):
        x = self.conv(x).mean((2, 3))

        return self.ffn(x)
