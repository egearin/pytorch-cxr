import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvm


class PositionalAttention(nn.Module):
    #Ref from SAGAN
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.zeros(1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : B x C x H x W
            returns :
                out : attention value + input feature
                attention: B x (HW) x (HW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = torch.zeros(1)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
 :
                x : B x C x H x W
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__()

        self.in_dim = 32 * 32
        self.out_dim = 32 * 32
        self.embed_dim = embed_dim * num_heads
        self.num_heads = num_heads

        self.in_proj = nn.Linear(self.in_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, self.out_dim)

        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, *args, **kwargs)

    def _proj_qkv(self, x):
        return self.in_proj(x).transpose(0, 1).chunk(3, dim=2)

    def forward(self, x):
        B, C, H, W = x.size()
        # input projection
        x = x.view(-1, C, H*W)
        q, k, v = self._proj_qkv(x)
        # calculate attention
        y, weights = self.attn(q, k, v)
        # output projection
        y = y.view(-1, self.num_heads, self.embed_dim // self.num_heads).sum(dim=1)
        y = self.out_proj(y)
        y = y.view(-1, C, H, W)
        return y


class Network(nn.Module):

    def __init__(self, in_dim=20, out_dim=14):
        super().__init__()

        self.attn = Attention(embed_dim=256, num_heads=4)

        self.main = tvm.resnext101_32x8d(pretrained=True)
        self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)

        # dim between layer2 and layer3 = B x 512 x 32 x 32
        self.main.layer3 = nn.Sequential(
            self.attn,
            self.main.layer3
        )

    def forward(self, x):
        x = self.main(x)
        return x


if __name__ == "__main__":
    model = Network()

    x = torch.rand((10, 20, 256, 256))
    out = model(x)
    print(out.shape)
