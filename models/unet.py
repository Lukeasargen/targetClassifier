import copy

import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F
from torchsummary import summary

import torchvision

# arguments
# input channels
# output channels
# filters, length is the number of downsamples
# pad, boolean, False does true convolution and crops the skip connections

# forward pass needs dropout during downsampling

# experiments
# padding vs true convolution
# activation on the transpose convolution
# maxpooling vs stride=2

# kernel size on the first conv
# kernel size on the downsample, changes the crop


def save_unet():
    pass

def load_unet():
    pass


class TwoConvBlock(nn.Module):
    """ Downsampling envolves two full convolutions with relu.
        The feature is reduced by 4 pixels. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO : batchnorm? or use bias?
        # paper did not use batchnorm
        modules = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        filters.insert(0, in_channels)
        blocks = [ TwoConvBlock(filters[i], filters[i+1]) for i in range(len(filters)-1) ]
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = []  # list of outputs from all TwoConvBlock downsampling
        for b in self.blocks:
            x = b(x)
            out.append(x)
            x = self.pool(x)
        return out 


class UpSample(nn.Module):
    def __init__(self, filters):
        super().__init__()
        filters.reverse()
        upconvs = [ nn.ConvTranspose2d(filters[i], filters[i+1], kernel_size=2, stride=2, bias=True) for i in range(len(filters)-1) ]
        self.upconvs = nn.ModuleList(upconvs)
        blocks = [ TwoConvBlock(filters[i], filters[i+1]) for i in range(len(filters)-1) ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        out = x
        for i in range(len(self.upconvs)-1):
            out = self.upconvs[i](out)
            temp = self.crop(skips[i], out)
            out = torch.cat([out, temp], dim=1)
            out = self.blocks[i](out)
        return out
    
    def crop(self, x, target):
        """ Takes x and crops it to size of target. 
            Assumes target is always smaller than x."""
        dh = ( x.shape[2] - target.shape[2] ) // 2
        dw = ( x.shape[3] - target.shape[3] ) // 2
        return x[:, :, dh:x.shape[2]-dh, dw:x.shape[3]-dw]


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, filters):
        super().__init__()
        self.model_args = {
            "in_channels": in_channels, "out_channels": out_channels,
            "filters": filters
        }
        self.down = DownSample(in_channels, filters.copy())
        self.up = UpSample(filters.copy())
        self.head = nn.Conv2d(in_channels=filters[-1], out_channels=out_channels,
                kernel_size=(1, 1), stride=1, padding=0, bias=True)

    def forward(self, x):
        skips = self.down(x)
        out = self.up(skips[::-1][0], skips[::-1][1:])
        out = self.head(out)
        return out


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 572
    in_channels = 3
    out_channels = 1
    filters = [16, 32, 64]


    filters = [64,128,256,512,1024]
    # down = DownSample(in_channels, filters)
    x = torch.randn(1, 3, 572, 572)
    # skip = down(x)
    # for c in skip:
    #     print("Down :", c.shape)

    # up = UpSample(filters)
    # x = torch.randn(1, 1024, 28, 28)
    # m = up(x, skip[::-1][1:])
    # print(m.shape)


    model = Unet(in_channels, out_channels, filters)

    out = model(x)

    print(out.shape)

    print(model)
    # summary(model.to(device), (in_channels, input_size, input_size))
