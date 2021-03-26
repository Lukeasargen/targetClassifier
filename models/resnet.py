from collections import OrderedDict

import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F
from torchsummary import summary


def save_resnet(model, save_path):
    data = {
        'model': model.state_dict(),
        'model_args': model.model_args,
    }
    torch.save(data, save_path)


def load_resnet(path, device):
    data = torch.load(path, map_location=torch.device(device))
    model = BasicResnet(**data['model_args']).to(device)
    model.load_state_dict(data['model'])
    return model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
            bottleneck=False, groups=1, width_per_group=None):
        super(ResidualBlock, self).__init__()

        if bottleneck:
            if width_per_group:  # TODO : I don't think this is right
                downscale_filters = int(out_channels * (width_per_group / 256.)) * groups
            else:
                downscale_filters = int(out_channels / 4)

            # print("downscale_filters :", downscale_filters)
            
            self.block = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=downscale_filters,
                    kernel_size=(1, 1), stride=1, padding=0, bias=False)),
                ('bn1', nn.BatchNorm2d(downscale_filters)),
                ('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ('conv2', nn.Conv2d(in_channels=downscale_filters, out_channels=downscale_filters,
                    kernel_size=(3, 3), stride=stride, padding=1, groups=groups, bias=False)),
                ('bn2', nn.BatchNorm2d(downscale_filters)),
                ('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ('conv3', nn.Conv2d(in_channels=downscale_filters, out_channels=out_channels,
                    kernel_size=(1, 1), stride=1, padding=0, bias=False)),
                ('bn3', nn.BatchNorm2d(out_channels)),
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(3, 3), stride=stride, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(out_channels)),
                ('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=(3, 3), stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(out_channels)),
            ]))
        
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()  # Identity layer
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BasicResnet(nn.Module):
    def __init__(self, in_channels, num_classes=1000, avgpool_size=None,
            filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], bottleneck=False,
            groups=1, width_per_group=None, last_conv=False):
        super(BasicResnet, self).__init__()
        assert (len(filters)-1)==len(blocks), "filters and blocks length do not match."
        self.num_blocks = len(blocks)

        self.model_args = {
            "in_channels": in_channels, "num_classes": num_classes,
            "avgpool_size": avgpool_size, "filters": filters,
            "blocks": blocks, "bottleneck": bottleneck, 
            "groups": groups, "width_per_group": width_per_group,
            "last_conv": last_conv
        }
        
        self.normalize = nn.BatchNorm2d(in_channels)  # Layer will be frozen without learnable parameters

        first_modules = [
            nn.Conv2d(in_channels=in_channels, out_channels=filters[0],
            kernel_size=(5, 5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        self.first_layer = nn.Sequential(*first_modules)

        for idx, num in enumerate(blocks):
            stride = 2
            if idx == 0: stride=1
            setattr(self, "Block"+str(idx), self._create_block(filters[idx], filters[idx+1], stride, blocks[idx], bottleneck, groups, width_per_group))

        # Remove last conv because each classifier head gets the averaged feature maps as input
        # Set last_conv=True to get a basic linear classifier
        last_modules = []
        if avgpool_size == None:
            last_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            last_modules.append(nn.AvgPool2d(avgpool_size))
        if last_conv:
            last_modules.append(nn.Conv2d(in_channels=filters[-1], out_channels=num_classes,
                                kernel_size=(1, 1), stride=1, padding=0, bias=True))
            last_modules.append(nn.Flatten())
        self.last_layer = nn.Sequential(*last_modules)

        """ https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.set_normalization()  # Freezes first layer, set to identity function
    
    def _create_block(self, in_channels, out_channels, stride, n, bottleneck, groups, width_per_group):
        layers = []
        for i in range(n):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, stride, bottleneck, groups, width_per_group))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, 1, bottleneck, groups, width_per_group))
        return nn.Sequential(*layers)

    def set_normalization(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.normalize.reset_parameters()
        self.normalize.running_mean = torch.tensor(mean, requires_grad=False, dtype=torch.float)
        self.normalize.running_var = torch.tensor([x**2 for x in std], requires_grad=False, dtype=torch.float)
        self.normalize.weight.requires_grad = False  # gamma
        self.normalize.bias.requires_grad = False   # beta
        self.normalize.running_mean.requires_grad = False  # mean
        self.normalize.running_var.requires_grad = False  # variance
        self.normalize.eval()

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        """ Replace pytorch train method in order to keep the normalization
            layer frozen. """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.normalize.eval()
        return self

    def forward(self, x, dropout=0.0):
        out = self.normalize.eval()(x)
        out = self.first_layer(out)
        for idx in range(self.num_blocks):
            out = eval("self.Block" + str(idx))(out)
            out = F.dropout(out, p=dropout)
        out = self.last_layer(out)
        return out


def resnet18(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2],
            last_conv=True)

def resnet34(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 64, 128, 256, 512], blocks=[3, 4, 6, 3],
            last_conv=True)

def resnet50(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 6, 3],
            bottleneck=True,
            last_conv=True)

def resnet101(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 23, 3],
            bottleneck=True,
            last_conv=True)

def resnet152(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 8, 36, 3],
            bottleneck=True,
            last_conv=True)

def resnext50_32x4d(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 6, 3],
            bottleneck=True, groups=32, width_per_group=4,
            last_conv=True)

def resnext101_32x8d(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 23, 3],
            bottleneck=True, groups=32, width_per_group=8,
            last_conv=True)

def wide_resnet50_2(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 6, 3],
            bottleneck=True, groups=1, width_per_group=128,
            last_conv=True)

def wide_resnet101_2(num_classes=1000):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[64, 256, 512, 1024, 2048], blocks=[3, 4, 23, 3],
            bottleneck=True, groups=1, width_per_group=128,
            last_conv=True)

def wide_resnet(num_classes=1000, l=2, k=8):
    return BasicResnet(in_channels=3, num_classes=num_classes,
            filters=[16, 16*k, 32*k, 64*k], blocks=[l, l, l],
            last_conv=True)

def wide_resnet16_8(num_classes=1000):
    return wide_resnet(num_classes, l=2, k=8)


def count_convs(module):
    c = 0
    if isinstance(module, nn.modules.conv._ConvNd):
        c +=1
    for child in module.children():
        c += count_convs(child)
    return c

def test_save_load(device):

    in_channels = 3
    input_size = 224
    
    x = torch.ones((1, in_channels, input_size, input_size)).to(device)

    m1 = resnet18(num_classes=6)
    m1.set_normalization(mean=[0.5, 0.4, 0.3], std=[0.1, 0.2, 0.3])
    m1.to(device)
    y1 = m1(x)
    print(y1, y1.size())

    path = 'runs/res_test.pth'
    save_resnet(m1, save_path=path)

    m2 = load_resnet(path, device)
    y2 = m2(x)
    print(y2, y2.size())


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 32
    in_channels = 3
    num_classes = 1000
    avgpool_size = None  # input_size/( + 2*(num_blocks - 1 + first_conv stride)
    filters = [16, 16, 32, 64]
    blocks = [2, 2, 2]
    bottleneck = False
    groups = 1
    width_per_group = None
    last_conv = True

    # model = BasicResnet(in_channels, num_classes, avgpool_size, filters, blocks,
    #         bottleneck, groups, width_per_group, last_conv)

    model = wide_resnet16_8(num_classes=10)
    model.set_normalization(mean=[0.5, 0.4, 0.3], std=[0.1, 0.2, 0.3])
    model.to(device)

    x = torch.ones((1, in_channels, input_size, input_size)).to(device)
    y = model(x)

    # summary(model, (in_channels, input_size, input_size))

    # print("Convolutional Layers:", count_convs(model))

    # test_save_load(device)
