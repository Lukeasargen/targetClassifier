import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, features):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=features)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(
            self.batchnorm1(
                self.conv1(x)))
        return self.relu2(
            self.batchnorm2(
                self.conv2(out)))


class old_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=16):
        super(old_UNet, self).__init__()

        self.encoder_block1 = UNetBlock(in_channels=in_channels,
                                        features=features)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block2 = UNetBlock(in_channels=features,
                                        features=features * 2)
        self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block3 = UNetBlock(in_channels=features * 2,
                                        features=features * 4)
        self.pool_layer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block4 = UNetBlock(in_channels=features * 4,
                                        features=features * 8)
        self.pool_layer4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck_layer = UNetBlock(in_channels=features * 8, features=features * 16)

        self.convT4 = nn.ConvTranspose2d(
            in_channels=features * 16, out_channels=features * 8,
            kernel_size=2, stride=2)

        self.decoder_block4 = UNetBlock(in_channels=features * 16, features=features * 8)
        self.convT3 = nn.ConvTranspose2d(
            in_channels=features * 8, out_channels=features * 4,
            kernel_size=2, stride=2)

        self.decoder_block3 = UNetBlock(in_channels=features * 8, features=features * 4)
        self.convT2 = nn.ConvTranspose2d(
            in_channels=features * 4, out_channels=features * 2,
            kernel_size=2, stride=2)

        self.decoder_block2 = UNetBlock(in_channels=features * 4, features=features * 2)
        self.convT1 = nn.ConvTranspose2d(
            in_channels=features * 2, out_channels=features,
            kernel_size=2, stride=2)

        self.decoder_block1 = UNetBlock(in_channels=features * 2, features=features)
        self.out_conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(
            self.pool_layer1(e1))
        e3 = self.encoder_block3(
            self.pool_layer2(e2))
        e4 = self.encoder_block4(
            self.pool_layer3(e3))
        bottleneck = self.bottleneck_layer(
            self.pool_layer4(e4))
        d4 = self.convT4(bottleneck)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder_block4(d4)
        d3 = self.convT3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder_block3(d3)
        d2 = self.convT2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder_block2(d2)
        d1 = self.convT1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder_block1(d1)
        return torch.sigmoid(self.out_conv(d1))


def test_unet(features, n_trials):
    model = old_UNet(in_channels=3, out_channels=1, features=features)
    n_params = 0
    for param in list(model.parameters()):
        nn_ = 1
        for s in list(param.size()):
            nn_ = nn_ * s
        n_params += nn_
    print('features = {} --> n_parameters = {}'.format(features, n_params))
    z = torch.zeros(size=(1, 3, 256, 416))
    from time import time
    with torch.no_grad():
        t_init = time()
        for _ in range(n_trials):
            model(z)
    print('seconds per input: {}'.format(round((time() - t_init) / n_trials, 4)))


if __name__ == "__main__":

    from torchsummary import summary

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    in_channels = 3
    out_channels = 1
    features = 16
    input_size = 400

    model = old_UNet(in_channels=in_channels, out_channels=out_channels, features=features)

    summary(model.to(device), (in_channels, input_size, input_size))

