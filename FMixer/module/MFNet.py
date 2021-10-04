r'''
Author       : PiKaChu_wcg
Date         : 2021-09-11 13:57:37
LastEditors  : PiKachu_wcg
LastEditTime : 2021-09-12 21:55:29
FilePath     : \毕设\module\MFNet.py
'''


import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    """
    the prenorm for the module
    we can choice the norm as the layernorm or the batchnorm

    """

    def __init__(self,  dim, fn, norm=nn.LayerNorm):
        super(PreNorm, self).__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ComplexFn(nn.Module):
    """
    wrap the model and make model can deal the complex number 
    """

    def __init__(self, fn):
        super(ComplexFn, self).__init__()
        self.fn = fn

    def forward(self, x):
        if x.dtype == torch.complex64:
            x.real, x.imag = self.fn(x.real), self.fn(x.imag)
        else:
            x=self.fn(x)
        return x


class FeedForward(nn.Module):
    """
    FeedForward block, it include [linear,gule,dropout,linear,dropout]
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net = ComplexFn(net)

    def forward(self, x):
        return self.net(x)


class FNetBlock(nn.Module):
    """
    fft the input the last two dimension
    """

    def __init__(self):
        super(FNetBlock, self).__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2)
        return x


class Mixer(nn.Module):
    '''
    change the order of the input dimension
    '''

    def __init__(self):
        super(Mixer, self).__init__()
        self.rerange = Rearrange("b c h w->b c h w")

    def forward(self, x):
        return self.rerange(x)


class MFBlock(nn.Module):
    def __init__(self, dim, mlp_dim=None, dropout=0.):
        super(MFBlock,self).__init__()
        self.net = nn.Sequential(
            FNetBlock(),
            ComplexFn(nn.LayerNorm(dim)),
            ComplexFn(PreNorm(dim, FeedForward(
                dim, mlp_dim, dropout=dropout))),
            Mixer(),
        )
        
    def forward(self ,x):
        return self.net(x)

class MFStage(nn.Module):
    def __init__(self,  channels, dim, mlp_dim=None, dropout=0.):
        super(MFStage, self).__init__()
        mlp_dim = mlp_dim if mlp_dim else dim*4
        self.net = nn.Sequential(
            MFBlock(dim,mlp_dim,dropout),
            ComplexFn(nn.Conv2d(channels, channels, 1, 1, bias=False)),
            MFBlock(dim, mlp_dim, dropout),
            ComplexFn(nn.Conv2d(channels, channels, 1, 1, bias=False)),
        )
    def forward(self, x):
        return self.net(x)


class MFNet(nn.Module):
    def __init__(self, num_layers,  channels, img_size, shrink, classes_num, in_channels=3, mlp_dim=None, dropout=0.):
        super(MFNet,self).__init__()
        dim = img_size//(2**shrink)
        mlp_dim = mlp_dim if mlp_dim else dim*4
        self.conv_embd = nn.ModuleList(
            [nn.Conv2d(in_channels, channels, 1, 1)])
        for _ in range(shrink):
            self.conv_embd.append(nn.Conv2d(channels, channels, 3, 2,1))
        self.net = nn.ModuleList([])
        for _ in range(num_layers):
            self.net.append(MFStage(channels, dim))
        self.head = nn.Linear(channels, classes_num)

    def forward(self, x):
        for l in self.conv_embd:
            x=l(x)
        for l in self.net:
            x = l(x)
        x = x.mean(dim=[-1, -2]).real
        x = self.head(x)
        return x


if __name__ == "__main__":
    import torchvision
    model = MFNet(1, 16, 224, 1, 10)
    train = torchvision.datasets.CIFAR10(root="F:\毕设\dataset", transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(224)]))
    t = train[0][0]
    t = t.view(1, *t.shape)
    print(model(t))
