import torch.nn as nn
import torch.nn.functional as F


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

class HourGlass(nn.Module):
    def __init__(self, n, dims, modules, make_pool_layer=nn.Sequential, **kwargs):
        super(HourGlass, self).__init__()
        self.n   = n
        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_layer(3, curr_dim, curr_dim, curr_mod, layer=residual, **kwargs)

        self.max1 = make_pool_layer()
        self.low1 = make_hg_layer(3, curr_dim, next_dim, curr_mod, layer=residual, **kwargs)
        self.low2 = HourGlass(n-1, dims[1:], modules[1:],
                              make_pool_layer=make_pool_layer, **kwargs) if self.n > 1 else \
            make_layer(3, next_dim, next_dim, next_mod, layer=residual, **kwargs)
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_mod, layer=residual, **kwargs)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = F.interpolate(low3, size=(x.size(-2), x.size(-1)))
        return self.merge(up1, up2)


class BackBone(nn.Module):
    def __init__(self, n=5, nstack=1, dims=(256, 256, 384, 384, 384, 512),
                 modules=(2,2,2,2,2,4), pre=None, cnv_dim=256):
        super(BackBone, self).__init__()
        self.nstack     = nstack
        curr_dim        = dims[0]

        self.pre  = nn.Sequential(convolution(7, 3, 128, stride=2),
                                  residual(3, 128, 256, stride=2)) if pre is None else pre

        self.hg   = HourGlass(n, dims, modules)
        self.cnv  = convolution(3, curr_dim, cnv_dim)

    def forward(self, x):
        """x: N x 3 x H x W """
        inter  = self.pre(x)
        hg_out = self.hg(inter)
        cnvout = self.cnv(hg_out)
        return cnvout

if __name__ == '__main__':
    import torch
    net = BackBone()
    print(net)
    x = torch.rand(2, 3, 259, 259)
    y = net(x)
    print(y.size())