
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_unet(model, save_path, mean, std):
    data = {
        'model': model.state_dict(),
        'model_args': model.model_args,
        'mean': mean,
        'std': std
    }
    print(f"saving unet to: {save_path}")
    torch.save(data, save_path)


def load_unet(path, device):
    data = torch.load(path, map_location=torch.device(device))
    model = UNet(**data['model_args']).to(device)
    model.load_state_dict(data['model'])
    return model, data


@torch.jit.script
def autocrop(decoder_layer: torch.Tensor, encoder_layer: torch.Tensor):
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        encoder_layer = encoder_layer[:, :,
                        ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                        ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                        ]
    return decoder_layer, encoder_layer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder_layer, encoder_layer):
        decoder_layer = self.up(decoder_layer)
        decoder_layer, encoder_layer = autocrop(decoder_layer, encoder_layer)
        x = torch.cat([encoder_layer, decoder_layer], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.model_args = {"in_channels": in_channels, "out_channels": out_channels, "filters": filters}

        self.down0 = DoubleConv(in_channels, filters)
        self.down1 = Down(filters, filters*2)
        self.down2 = Down(filters*2, filters*4)
        self.down3 = Down(filters*4, filters*8)
        self.down4 = Down(filters*8, filters*16)
        self.up1 = Up(filters*16, filters*8)
        self.up2 = Up(filters*8, filters*4)
        self.up3 = Up(filters*4, filters*2)
        self.up4 = Up(filters*2, filters)
        self.out = nn.Conv2d(filters, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.down0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.out(x))


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 400
    in_channels = 3
    out_channels = 1
    filters = 16

    model = UNet(in_channels, out_channels, filters).to(device)

    # x = torch.randn(1, in_channels, input_size, input_size).to(device)
    # out = model(x)
    # print(out.shape, torch.min(out).data, torch.max(out).data)

    from torchsummary import summary
    summary(model.to(device), (in_channels, input_size, input_size))

