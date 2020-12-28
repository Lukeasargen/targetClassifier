from collections import OrderedDict
import torch
import torch.nn as nn  # Building the model
from torch.autograd import Variable
from torchsummary import summary


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
                ('conv2', nn.Conv2d(in_channels=downscale_filters, out_channels=downscale_filters,
                    kernel_size=(3, 3), stride=stride, padding=1, groups=groups, bias=False)),
                ('bn2', nn.BatchNorm2d(downscale_filters)),
                ('conv3', nn.Conv2d(in_channels=downscale_filters, out_channels=out_channels,
                    kernel_size=(1, 1), stride=1, padding=0, bias=False)),
                ('bn3', nn.BatchNorm2d(out_channels)),
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(3, 3), stride=stride, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(out_channels)),
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
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BasicResnet(nn.Module):
    def __init__(self, in_channels, out_features, avgpool_size,
            filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], bottleneck=False,
            groups=1, width_per_group=None):
        super(BasicResnet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.avgpool_size = avgpool_size
        self.filters = filters
        self.blocks = blocks
        self.bottleneck = bottleneck
        self.groups = groups
        self.width_per_group = width_per_group
        self.num_blocks = len(blocks)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters[0],
                kernel_size=(7, 7), stride=2, padding=3, bias=True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )

        assert (len(filters)-1)==len(blocks), "filters and blocks length do not match."
        for idx, num in enumerate(blocks):
            stride = 2
            if idx == 0: stride=1
            setattr(self, "Block"+str(idx), self._create_block(filters[idx], filters[idx+1], stride, blocks[idx], bottleneck, groups, width_per_group))
        self.avgpool = nn.AvgPool2d(avgpool_size)

        # Remove last conv because each classifier head gets the averaged feature maps as input
        # self.last_conv = nn.Conv2d(in_channels=filters[4], out_channels=out_features,
        #     kernel_size=(1, 1), stride=1, padding=0, bias=True)

        """ https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _create_block(self, in_channels, out_channels, stride, n, bottleneck, groups, width_per_group):
        layers = []
        for i in range(n):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, stride, bottleneck, groups, width_per_group))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, 1, bottleneck, groups, width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x, dropout_conv=0.0):
        drop_conv_layer = nn.Dropout2d(p=dropout_conv)
        out = self.conv1(x)
        out = drop_conv_layer(out)
        for idx in range(self.num_blocks):
            out = eval("self.Block" + str(idx))(out)
            out = drop_conv_layer(out)
        out = self.avgpool(out)
        # out = self.last_conv(out)
        return out


class MultiTaskTargetNet(nn.Module):
    def __init__(self, basemodel, backbone_features, num_classes):
        super(MultiTaskTargetNet, self).__init__()
        self.basemodel = basemodel
        self.backbone_features = backbone_features
        self.num_classes = num_classes
        # make the multi label heads
        for index, num in enumerate(num_classes):
            setattr(self, "ClassifierHead_"+str(index), self._make_classifier(backbone_features, num))
    
    def _make_classifier(self, in_features, num_classes):
        head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_features, out_channels=num_classes,
                kernel_size=(1, 1), stride=1, padding=0, bias=True)),
            # ('conv2', nn.Conv2d(in_channels=2*num_classes, out_channels=num_classes,
            #     kernel_size=(1, 1), stride=1, padding=0, bias=True)),
            ('flatten', nn.Flatten())
        ]))
        return head

    def forward(self, x, dropout_conv=0.0):
        x = self.basemodel.forward(x, dropout_conv)
        # forward the label heads
        outs = list()
        for index in range(len(self.num_classes)):
            out = eval("self.ClassifierHead_" + str(index))(x)
            outs.append(out)
        return outs

def BuildMultiTaskTargetNet(backbone_features, num_classes, in_channels, avgpool_size,
        filters=[64, 64, 128, 256, 512], blocks=[2, 2, 2, 2], bottleneck=False,
        groups=1, width_per_group=None):
    """ 
        in_channels : size of the tensor in the channel dimension\n
        backbone_features : number of features generated by the backbone
        and passed to the classifier heads. The classifier head designs expects
        [-1, backbone_features, 1, 1] input. Each head is made of a single 1x1
        convultion and a flatten layer.\n
        num_classes : list of the length of each multi task classifier head\n
    """
    m = BasicResnet(in_channels, backbone_features, avgpool_size, filters, blocks,
                bottleneck, groups, width_per_group)
    return MultiTaskTargetNet(m, backbone_features, num_classes)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_size = 32
    in_channels = 3
    backbone_features = 64
    avgpool_size = 4  # input_size/(2 + 2*(num_blocks-1)
    filters = [16, 16, 32, 64]
    blocks = [2, 2, 2]
    bottleneck = False
    groups = 1
    width_per_group = None
    num_classes = [16, 13, 34, 10, 10]

    model = BuildMultiTaskTargetNet(backbone_features, num_classes, in_channels, avgpool_size,
                filters, blocks, bottleneck, groups, width_per_group)
    # print(model)
    summary(model.to(device), (in_channels, input_size, input_size))

    exit()

    """
    filters = [64, 64, 128, 256, 512]
    bottleneck = False
    resnet18
    blocks = [2, 2, 2, 2]
    resnet34
    blocks = [3, 4, 6, 3]

    filters = [64, 256, 512, 1024, 2048]
    bottleneck = True
    resnet50 
    blocks = [3, 4, 6, 3]
    resnet101
    blocks = [3, 4, 23, 3]
    resnet152
    blocks = [3, 8, 36, 3]
    resnext50_32x4d
    blocks = [3, 4, 6, 3]
    groups = 32
    width_per_group = 4
    resnext101_32x8d
    blocks = [3, 4, 23, 3]
    groups = 32
    width_per_group = 8
    """

    # fake training loop

    print(label_weights)

    batch = 8
    x = torch.rand((batch, in_channels, input_size, input_size)).to(device)
    target = torch.hstack([torch.randint(0, n, size=(batch,1)).to(device) for n in num_classes])

    print("target :", target, target.shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    def scalar_loss(output, target):
        loss_func = nn.L1Loss()
        # loss_func = nn.MSELoss()
        return loss_func(output.squeeze(), target)

    epochs = 4
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        
        # Computing the loss and training metrics
        label_loss = Variable(torch.FloatTensor(len(num_classes))).zero_().to(device)
        for i in range(len(num_classes)):
            y = target
            if target.ndim != 1:
                y = target[:, i]
            if output[i].shape[1] == 1:  # output is a scalar not a class distribution
                label_loss[i] = scalar_loss(output[i], y)
            else:  # class cross entropy loss
                # TODO : class weights
                label_loss[i] = criterion(output[i], y)
    
            # ll = label_loss.detach().clone()
            # label_weights = (1-torch.exp(ll)/torch.exp(ll).sum())*ll

        # TODO : label weights
        batch_loss = (label_weights*label_loss).sum()
        batch_loss.backward()
        optimizer.step()

        print("label_loss :", label_loss)
        # print("batch_loss :", batch_loss.item())
        # print("loss 2  :", label_loss.sum().item())
        print("label_weights :", label_weights)
        print("Epoch {} Loss={}".format(epoch, batch_loss.item()))




