r'''
Author       : PiKaChu_wcg
Date         : 2021-09-09 19:27:14
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-09 21:06:15
FilePath     : \毕设\module\fnet.py
'''
import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # todo the complex operation
        x.real=self.net(x.real)
        x.imag=self.net(x.imag)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)+x


class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x


class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential([])
        for _ in range(depth):
            self.layers.append(nn.Sequential([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        self.layers(x)
        return x
