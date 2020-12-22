from collections import OrderedDict
import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F  # Loss functions and convolutions
from torch.autograd import Variable
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
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

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TargetNet(nn.Module):
    def __init__(self, in_channels, out_features, dropout_conv=0.0):
        super(TargetNet, self).__init__()
        conv1_filters = 16
        block1_filters = 16
        block2_filters = 32
        block3_filters = 64
        block4_filters = 128
        conv_out_size = 4  # calculated, input_size/16
        blocks = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_filters,
            kernel_size=(7, 7), stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout2d(p=dropout_conv)
        self.block1 = self._create_block(conv1_filters, block1_filters, stride=1, n=blocks[0])
        self.block2 = self._create_block(block1_filters, block2_filters, stride=2, n=blocks[1])
        self.block3 = self._create_block(block2_filters, block3_filters, stride=2, n=blocks[2])
        self.block4 = self._create_block(block3_filters, block4_filters, stride=2, n=blocks[3])
        self.last_conv = nn.Conv2d(in_channels=block4_filters, out_channels=out_features,
            kernel_size=(1, 1), stride=1, padding=0, bias=True)
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
        out = self.block3(out)
        out = self.block4(out)
        out = self.avgpool2d(out)
        out = self.last_conv(out)
        return out


class MultiLabelTargetNet(nn.Module):
    def __init__(self, basemodel, backbone_features, num_classes):
        super(MultiLabelTargetNet, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        # make the multi label heads
        for index, num in enumerate(num_classes):
            setattr(self, "ClassifierHead_" + str(index), self._make_classifier(backbone_features, num))
    
    def _make_classifier(self, in_features, num_classes):
        head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_features, out_channels=num_classes,
                kernel_size=(1, 1), stride=1, padding=0, bias=True)),
            # ('conv2', nn.Conv2d(in_channels=2*num_classes, out_channels=num_classes,
            #     kernel_size=(1, 1), stride=1, padding=0, bias=True)),
            ('flatten', nn.Flatten())
        ]))
        return head

    def forward(self, x):
        x = self.basemodel.forward(x)
        # forward the label heads
        outs = list()
        for index in range(len(self.num_classes)):
            out = eval("self.ClassifierHead_" + str(index))(x)
            outs.append(out)
        return outs

def BuildMultiLabelTargetNet(in_channels, backbone_features, num_classes, dropout_conv=0.0):
    """ 
        in_channels : size of the tensor in the channel dimension\n
        backbone_features : number of features generated by the backbone and passed to the classifier heads\n
        num_classes : list of the length of each multi label classifier head\n
    """
    m = TargetNet(in_channels, backbone_features, dropout_conv)
    return MultiLabelTargetNet(m, backbone_features, num_classes)


if __name__ == "__main__":
    device = 'cuda'
    input_size = 64
    backbone_features = 128
    in_channels = 3

    num_classes = [13, 10, 34, 10]
    label_weights = torch.FloatTensor([1.0]*len(num_classes))
    label_weights = torch.FloatTensor([1.0, 1.0, 10.0, 1.0]).to(device)
    model = BuildMultiLabelTargetNet(in_channels, backbone_features, num_classes).to(device)
    # print(model)
    # summary(model, (in_channels, input_size, input_size))

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




