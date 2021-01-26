
import torch
import torch.nn as nn  # Building the model


class Net(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.normalize = nn.BatchNorm2d(in_channels)  # Layer will be frozen without learnable parameters
        self.normalize.reset_parameters()
        self.normalize.eval()

        self.set_normalization()
        
    def set_normalization(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.normalize.weight.requires_grad = False  # gamma
        self.normalize.bias.requires_grad = False   # beta
        self.normalize.running_mean.requires_grad = False  # stat, mean
        self.normalize.running_var.requires_grad = False  # stat, variance

        self.normalize.weight = torch.nn.Parameter( torch.ones(3), requires_grad=False )
        self.normalize.bias = torch.nn.Parameter( torch.zeros(3), requires_grad=False )
        self.normalize.running_mean = torch.nn.Parameter( torch.tensor(mean), requires_grad=False )
        self.normalize.running_var = torch.nn.Parameter( torch.tensor([x**2 for x in std]), requires_grad=False )

        print("\nweight :", self.normalize.weight)
        print("bias :", self.normalize.bias)  
        print("running_mean :", self.normalize.running_mean)
        print("running_var :", self.normalize.running_var)

    def forward(self, x):
        print("pre :", x)
        x = self.normalize(x)
        print("post :", x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size = 2
in_channels = 3

model = Net(in_channels)

model.set_normalization(mean=[0.5, 0.4, 0.3], std=[0.1, 0.2, 0.3])

x = torch.rand((1, in_channels, input_size, input_size)).to(device)

model = model.to(device)

y = model(x)

