import os  # Get files
import time  # Time training
import numpy as np  # Image transforms, random selection
import matplotlib.pyplot as plt  # Display lr finder
import torch  # Tensor library
import torch.nn as nn  # Building the model, loss func
import torch.optim as optim  # Optimization and schedulers
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision  # Datasets
import torchvision.transforms as T  # Image processing
from torch.autograd import Variable

from generate_samples import TargetGenerator
from model import TargetNet

# Data loader class
class TargetDataset(Dataset):
    def __init__(self, transforms, input_size, target_size, length):
        self.transforms = transforms
        self.gen = TargetGenerator()
        self.length = length
        self.input_size = input_size
        self.target_size = target_size

    def get_target_size(self):
        return int(np.random.uniform(0.9*target_size, target_size))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = self.gen.draw_target(self.input_size, self.get_target_size())
        return self.transforms(x), torch.tensor(list(y.values())).squeeze()


if __name__ == "__main__" and '__file__' in globals():

    MANUAL_SEED = 101
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)


    # Model Config
    input_size = 64
    in_channels = 3
    # num classes is defined after dataset is made

    # Training Hyperparameters
    num_epochs = 14

    batch_size = 128
    num_workers = 0
    shuffle = False
    drop_last = True

    dropout_conv = 0.0
    base_lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4

    lr_step_size = 7
    lr_step_gamma = 0.1

    # Dataset config
    class_type_idx = 0
    target_size = 48
    train_size = 4096
    test_size = 1024

    set_mean = [0.274, 0.345, 0.451]
    set_std = [0.319, 0.350, 0.429]

    train_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
    ])

    train_dataset = TargetDataset(
        transforms=train_transforms,
        input_size=input_size,
        target_size=target_size,
        length=train_size)
    train_loader = DataLoader(
        dataset=train_dataset
        ,batch_size=batch_size
        ,shuffle=shuffle
        ,num_workers=num_workers
        ,drop_last=drop_last)

    test_dataset = TargetDataset(
        transforms=test_transforms,
        input_size=input_size,
        target_size=target_size,
        length=test_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    # TODO : fixed for multiclass
    num_classes = test_dataset.gen.num_classes[class_type_idx]


    exit()

    # Time dataloader
    # import time
    # for i in range(9):
    #     train_loader = DataLoader(
    #         dataset=train_dataset
    #         ,batch_size=batch_size
    #         ,shuffle=shuffle
    #         ,num_workers=i
    #         ,drop_last=drop_last)
    #     t0 = time.time()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         pass
    #     print("{:.6f} seconds with {} workers.".format(time.time()-t0, i))
    # exit()

    # Check for cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = TargetNet(in_channels, num_classes, dropout_conv).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=base_lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)

    # Training
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    def train(epoch, model, optimizer, dataloader, criterion, scheduler, train=False):

        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            # data [B, C, W, H]
            # target [B, N] n is index of class

            # print("data :", data.shape)
            # print("target 1:", target, target.shape)
            target = target[:, class_type_idx]  # 0 is color
            # print("target 2:", target, target.shape)

            output = model(data)
            # print("output 1:", output.shape)

            # Flatten the 1x1 maps, [B, C]
            out = output.view(output.size(0), -1)
            pred = torch.max(out, 1)[1]  # second tensor is indicies
            # print("pred:", pred, pred.shape)

            correct += (pred == target).sum()
            total += data.size(0)

            output = torch.squeeze(output)  # flatten the last conv
            # print("output 2:", output.shape)

            # Computing the loss
            # loss_func = nn.BCEWithLogitsLoss()
            # loss2 = loss_func() 

            loss = criterion(output, target)
            # print("loss :", loss)
            total_loss += loss.item()
            # print("total_loss :", total_loss)

            # Computing the updated weights of all the model parameters
            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        unit_loss = total_loss / len(dataloader)
        acc = 100 * correct / total

        print("EPOCH {}. train={}. Accuracy={:.2f}%. Loss={:.4f}".format(epoch, train, acc, unit_loss))

        if train:
            train_loss.append(unit_loss)
            train_accuracy.append(acc)
        else:
            valid_loss.append(unit_loss)
            valid_accuracy.append(acc)


    for epoch in range(num_epochs):
        train(epoch, model, optimizer, train_loader, criterion, scheduler, train=True)
        train(epoch, model, optimizer, test_loader, criterion, scheduler, train=False)


    # exit()
    # Display loss graphs
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # plt.plot(range(1, len(train_loss)+1), train_loss, 'b-', label='training loss')
    # plt.plot(range(1, len(valid_loss)+1), valid_loss, 'g-', label='validation loss')
    plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'r-', label='training accuracy')
    plt.plot(range(1, len(valid_accuracy)+1), valid_accuracy, 'y-', label='validation accuracy')
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('loss', fontsize=16)

    ax = plt.gca()
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.show()
