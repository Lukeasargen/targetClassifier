from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from layers import get_act
except:
    try:
        from .layers import get_act
    except Exception as e:
        print(e)


def save_unet(model, save_path):
    data = {
        'model': model.state_dict(),
        'model_args': model.model_args,
    }
    print(f"saving unet to: {save_path}")
    torch.save(data, save_path)


def load_unet(path, device):
    data = torch.load(path, map_location=torch.device(device))
    model = UNet(**data['model_args']).to(device)
    model.load_state_dict(data['model'])
    return model


@torch.jit.script
def autocrop(encoder_features: torch.Tensor, decoder_features: torch.Tensor):
    """ Center crop the encoder down to the size of the decoder """
    if encoder_features.shape[2:] != decoder_features.shape[2:]:
        ds = encoder_features.shape[2:]
        es = decoder_features.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        encoder_features = encoder_features[:, :,
                        ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                        ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                        ]
    return encoder_features, decoder_features


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None):
        super().__init__()
        act_func = get_act(activation)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, stride, padding, activation)
        )

    def forward(self, x: torch.Tensor):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=None):
        super().__init__()
        mid_channels = mid_channels if mid_channels else 2*out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(mid_channels, out_channels, activation=activation)

    def forward(self, encoder_features: torch.Tensor, decoder_features: torch.Tensor):
        decoder_features = self.up(decoder_features)
        encoder_features, decoder_features = autocrop(encoder_features, decoder_features)
        x = torch.cat([encoder_features, decoder_features], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, filters, activation):
        super(UNetEncoder, self).__init__()
        self.down00 = DoubleConv(in_channels, filters[0], activation=activation)
        self.down10 = Down(filters[0], filters[1], activation=activation)
        self.down20 = Down(filters[1], filters[2], activation=activation)
        self.down30 = Down(filters[2], filters[3], activation=activation)
        self.down40 = Down(filters[3], filters[4], activation=activation)

    def forward(self, x: torch.Tensor):
        x00 = self.down00(x)
        x10 = self.down10(x00)
        x20 = self.down20(x10)
        x30 = self.down30(x20)
        x40 = self.down40(x30)
        return [x00, x10, x20, x30, x40]


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, filters, activation):
        super(UNetDecoder, self).__init__()
        self.up1 = Up(filters[4], filters[3], activation=activation)
        self.up2 = Up(filters[3], filters[2], activation=activation)
        self.up3 = Up(filters[2], filters[1], activation=activation)
        self.up4 = Up(filters[1], filters[0], activation=activation)
        self.out = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        x00, x10, x20, x30, x40 = features
        x = self.up1(x30, x40)
        x = self.up2(x20, x)
        x = self.up3(x10, x)
        x = self.up4(x00, x)
        return self.out(x)


class UNetNestedDecoder(nn.Module):
    def __init__(self, out_channels, filters, activation, deep=False):
        super(UNetNestedDecoder, self).__init__()
        self.deep = deep
        # L1
        self.up01 = Up(filters[1], filters[0], filters[0]*2, activation=activation)
        # L2
        self.up11 = Up(filters[2], filters[1], filters[1]*2, activation=activation)
        self.up02 = Up(filters[1]*2, filters[0], filters[0]*3, activation=activation)
        # L3
        self.up21 = Up(filters[3], filters[2], filters[2]*2, activation=activation)
        self.up12 = Up(filters[2]*2, filters[1], filters[1]*3, activation=activation)
        self.up03 = Up(filters[1]*3, filters[0], filters[0]*4, activation=activation)
        # L4
        self.up31 = Up(filters[4], filters[3], filters[3]*2, activation=activation)
        self.up22 = Up(filters[3]*2, filters[2], filters[2]*3, activation=activation)
        self.up13 = Up(filters[2]*3, filters[1], filters[1]*4, activation=activation)
        self.up04 = Up(filters[1]*4, filters[0], filters[0]*5, activation=activation)
        # Deep supervision
        if self.deep:
            self.ds01 = nn.Conv2d(filters[0]*2, out_channels, kernel_size=1)
            self.ds02 = nn.Conv2d(filters[0]*3, out_channels, kernel_size=1)
            self.ds03 = nn.Conv2d(filters[0]*4, out_channels, kernel_size=1)
        # Out
        self.out = nn.Conv2d(filters[0]*5, out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        x00, x10, x20, x30, x40 = features
        # L1
        x01 = torch.cat([x00, self.up01(x00, x10)], dim=1)
        # L2
        x11 = torch.cat([x10, self.up11(x10, x20)], dim=1)
        x02 = torch.cat([x01, self.up02(x01, x11)], dim=1)
        # L3
        x21 = torch.cat([x20, self.up21(x20, x30)], dim=1)
        x12 = torch.cat([x11, self.up12(x11, x21)], dim=1)
        x03 = torch.cat([x02, self.up03(x02, x12)], dim=1)
        # L4
        x31 = torch.cat([x30, self.up31(x30, x40)], dim=1)
        x22 = torch.cat([x21, self.up22(x21, x31)], dim=1)
        x13 = torch.cat([x12, self.up13(x12, x22)], dim=1)
        x04 = torch.cat([x03, self.up04(x03, x13)], dim=1)
        out = self.out(x04)
        # Deep supervision
        if self.deep:
            l1 = self.ds01(x01)
            l2 = self.ds02(x02)
            l3 = self.ds03(x03)
            out = (l1 + l2 + l3 + out) / 4.0
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, model_type='unet', filters=16, 
                activation='relu', mean=[0,0,0], std=[1,1,1], input_size=None):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.filters = filters
        self.input_size = input_size
        self.model_args = {"in_channels": in_channels, "out_channels": out_channels,  # mean and std are saved as parameters
            "model_type": model_type, "filters": filters, "activation": activation} 
        if type(filters) == int:
            filters = [filters, filters*2, filters*4, filters*8, filters*16]
        assert len(filters)==5

        self.encoder = UNetEncoder(in_channels, out_channels, filters, activation)

        assert model_type in ['unet', 'unet_nested', 'unet_nested_deep']
        if model_type == 'unet':
            self.decoder = UNetDecoder(out_channels, filters, activation)
        elif model_type == 'unet_nested':
            self.decoder = UNetNestedDecoder(out_channels, filters, activation)
        elif model_type == 'unet_nested_deep':
            self.decoder = UNetNestedDecoder(out_channels, filters, activation, deep=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if activation == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif activation == 'leaky_relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                else:
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.normalize = nn.BatchNorm2d(in_channels)  # Layer will be frozen without learnable parameters
        self.set_normalization(mean, std)

    def set_normalization(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.normalize.reset_parameters()
        self.normalize.running_mean = torch.tensor(mean, requires_grad=False, dtype=torch.float)
        self.normalize.running_var = torch.tensor([x**2 for x in std], requires_grad=False, dtype=torch.float)
        self.normalize.weight.requires_grad = False  # gamma
        self.normalize.bias.requires_grad = False   # beta
        self.normalize.running_mean.requires_grad = False  # mean
        self.normalize.running_var.requires_grad = False  # variance
        self.normalize.eval()

    @torch.jit.export
    def freeze_norm(self):
        self.normalize.training = False

    def forward(self, x):
        x = self.normalize(x)
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
    
    @torch.jit.export
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def train_test(model, device):
    batch = 2
    model.to(device).train()
    model.freeze_norm()
    x = torch.randn(batch, model.in_channels, 64, 64).to(device)
    y = torch.ones(batch, model.out_channels, 64, 64).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    for i in range(10):
        opt.zero_grad()
        logits = model(x)
        loss = 0.0
        loss += nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        opt.step()
        print(loss.item())


def profile_model(model, x):
    warmup = 3
    model.eval()
    for i in range(warmup):
        out = model.predict(x)
    import torch.autograd.profiler as profiler
    with profiler.profile(use_cuda=True, profile_memory=True) as prof:
        out = model.predict(x)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=16))
    print(torch.cuda.memory_summary())


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    in_channels = 3
    out_channels = 1
    model_type = "unet_nested_deep"  # unet, unet_nested, unet_nested_deep
    filters = 8
    activation = "relu"  # relu, leaky_relu, silu, mish

    batch_size = 1
    input_size = 192

    model = UNet(in_channels, out_channels, model_type, filters, activation, input_size=input_size).to(device)

    # train_test(model, device)

    x = torch.randn(batch_size, in_channels, input_size, input_size).to(device)

    model.train()
    model.freeze_norm()
    logits = model(x)
    print("logits :", logits.shape, torch.min(logits).item(), torch.max(logits).item())
    # predict = model.predict(x)
    # print("predict :", predict.shape, torch.min(predict).item(), torch.max(predict).item())
    
    # scripted = torch.jit.script(model)
    # scripted.save("runs/scripted_unet.pt")
    # scripted.train()
    # scripted.freeze_norm()
    # logits = scripted(x)
    # print("logits :", logits.shape, torch.min(logits).item(), torch.max(logits).item())
    # predict = scripted.predict(x)
    # print("predict :", predict.shape, torch.min(predict).item(), torch.max(predict).item())

    # profile_model(model, x)

    # from torchviz import make_dot
    # dot = make_dot(model(x), params=dict(list(model.named_parameters()) + [('x', x)]))
    # dot.format = 'png'
    # dot.render(f'images/{model_type}_graph')

    # save_unet(model, save_path="runs/unet_test.pth")
    # model2 = load_unet("runs/unet_test.pth", device=device)
    # out = model2(x)
    # print(out.shape, torch.min(out).item(), torch.max(out).item())


    # from pytorch_model_summary import summary  # git clone https://github.com/amarczew/pytorch_model_summary.git
    # summary(model,
    #         torch.zeros(1, in_channels, input_size, input_size).to(device),
    #         batch_size=1,
    #         show_input=False,  # Input shape or output shape
    #         show_hierarchical=False, 
    #         print_summary=True,  # calls print before returning
    #         max_depth=None,  # None searchs max depth
    #         show_parent_layers=False
    #         )
