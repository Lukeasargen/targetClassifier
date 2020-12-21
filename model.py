import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F  # Loss functions and convolutions
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.relu = nn.ReLU()
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()  # Identity layer
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TargetNet(nn.Module):
    def __init__(self, in_channels, out_features, dropout_conv=0.0):
        super(TargetNet, self).__init__()  # Base module

        conv1_filters = 32
        block1_filters = 32
        block2_filters = 32
        conv_out_size = 8

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=conv1_filters, kernel_size=(7, 7),
            stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout2d(p=dropout_conv)
        self.block1 = self._create_block(conv1_filters, block1_filters, stride=2, n=1)
        self.block2 = self._create_block(block1_filters, block2_filters, stride=2, n=1)
        self.last_conv = nn.Conv2d(
            in_channels=block2_filters, out_channels=out_features,
            kernel_size=(1, 1), stride=1, padding=0, bias=True
        )

        self.avgpool2d = nn.AvgPool2d(conv_out_size)
    
    def _create_block(self, in_channels, out_channels, stride, n):
        layers = []
        for i in range(n):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, stride))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.dropout_conv(out)
        out = self.block1(out)
        out = self.dropout_conv(out)
        out = self.block2(out)
        out = self.avgpool2d(out)
        out = self.last_conv(out)
        # flatten to features
        return out


class MultiLabelTargetNet(nn.Module):
    def __init__(self, basemodel, in_features, num_classes):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        # make the multi label heads
    
    def forward(self, x):
        x = self.basemodel.forward(x)
        # forward the label heads


if __name__ == "__main__":
    device = 'cuda'
    input_size = 64
    num_classes = 4
    in_channels = 3
    model = TargetNet(num_classes, in_channels).to(device)
    summary(model, (in_channels, input_size, input_size))
