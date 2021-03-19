import copy

import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F
from torchsummary import summary


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
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=(3, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        filters.insert(0, in_channels)
        self.blocks = nn.ModuleList([TwoConvBlock(filters[i], filters[i+1]) for i in range(len(filters)-1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
            x = self.pool(x)
        return out


class UpSample(nn.Module):
    def __init__(self, filters):
        super().__init__()
        filters.reverse()
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(filters[i], filters[i+1], kernel_size=2, stride=2, bias=True) for i in range(len(filters)-1)])
        self.blocks = nn.ModuleList([TwoConvBlock(filters[i], filters[i+1]) for i in range(len(filters)-1)])

    def forward(self, x, skips):
        out = x
        for i in range(len(self.upconvs)):
            # print(i, "before up :", out.shape)
            out = self.upconvs[i](out)
            # print(i, "after up :", out.shape)
            out = self.crop_and_cat(out, skips[i])
            # print(i, "after cat :", out.shape)
            out = self.blocks[i](out)
            # print(i, "after block :", out.shape)
        return out
    
    def crop_and_cat(self, upsample, skip):
        """ Takes upsample and crops it to size of skip. 
            Assumes skip is always smaller than upsample."""
        # print("upsample.shape :", upsample.shape)
        # print("skip.shape :", skip.shape)
        dh = ( skip.shape[2] - upsample.shape[2] ) // 2
        dw = ( skip.shape[3] - upsample.shape[3] ) // 2
        # print("dh :", dh)
        # print("dw :", dw)
        crop = skip[:, :, dh:skip.shape[2]-dh, dw:skip.shape[3]-dw]
        # print("crop.shape :", crop.shape)
        out = torch.cat([crop, upsample], dim=1)
        # print("out.shape :", out.shape)
        return out


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, filters):
        super().__init__()
        self.model_args = {
            "in_channels": in_channels, "out_channels": out_channels,
            "filters": filters
        }
        self.down = DownSample(in_channels, filters.copy())
        self.up = UpSample(filters.copy())
        self.head = nn.Conv2d(in_channels=filters[0], out_channels=out_channels,
                kernel_size=(1, 1), stride=1, padding=0, bias=True)

    def forward(self, x):
        out = None
        skips = self.down(x)
        # for i, x in enumerate(skips[::-1]):
        #     print(i, x.shape)
        out = self.up(skips[::-1][0], skips[::-1][1:])
        out = self.head(out)
        return out


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 828
    in_channels = 3
    out_channels = 1
    filters = [64,128,256,512,1024]

    model = Unet(in_channels, out_channels, filters).to(device)
    
    model.train()
    
    x = torch.randn(1, in_channels, input_size, input_size).to(device)
    out = model(x)
    print(out.shape)

    # summary(model.to(device), (in_channels, input_size, input_size))

