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
from model import BuildMultiLabelTargetNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Data loader class
class TargetDataset(Dataset):
    def __init__(self, transforms, input_size, target_size, length, scale, rotation):
        self.transforms = transforms
        self.gen = TargetGenerator()
        self.length = length
        self.input_size = input_size
        self.target_size = target_size
        self.scale = scale
        self.rotation = rotation

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = self.gen.draw_target(self.input_size, self.target_size, self.scale, self.rotation)
        return self.transforms(x), torch.tensor(list(y.values()), dtype=torch.long).squeeze()


def get_correct(output, target):
    pred = torch.max(output, 1)[1]  # second tensor is indicies
    correct = (pred == target).sum().item()
    return correct

def scalar_loss(output, target):
    """ Loss function when the output is a scalar (not a class distribution). """
    loss_func = nn.L1Loss()
    # loss_func = nn.MSELoss()
    return loss_func(output.squeeze(), target)

if __name__ == "__main__" and '__file__' in globals():
    MANUAL_SEED = 104
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)
    # Model Config
    input_size = 64
    in_channels = 3
    backbone_features = 128
    # num classes is defined after dataset is made
    # Training Hyperparameters
    num_epochs = 40
    validate = False
    batch_size = 256
    num_workers = 8
    shuffle = False
    drop_last = True
    dropout_conv = 0.0
    base_lr = 1e-5
    momentum = 0.9
    weight_decay = 4e-5
    lr_milestones = [120, 160]
    lr_gamma = 0.1
    # Dataset config
    target_size = 60
    scale = (1.0, 1.0)
    rotation = False
    train_size = 16384
    test_size = 1024
    set_mean = [0.248, 0.194, 0.171]
    set_std = [0.313, 0.261, 0.253]

    train_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
    ])

    train_dataset = TargetDataset(transforms=train_transforms, input_size=input_size,
        target_size=target_size, length=train_size, scale=scale, rotation=rotation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size
        ,shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    test_dataset = TargetDataset(transforms=test_transforms, input_size=input_size,
        target_size=target_size, length=test_size, scale=scale, rotation=rotation)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    num_classes = test_dataset.gen.num_classes
    
    label_weights = test_dataset.gen.label_weights
    if label_weights == None: label_weights = [1.0]*len(num_classes)

    print("num_classes :", num_classes)
    print("label_weights :", label_weights)

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
    print('device :', device)
    model = BuildMultiLabelTargetNet(in_channels, backbone_features, num_classes, dropout_conv).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=base_lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None # optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Training
    label_weights = torch.FloatTensor(label_weights).to(device)
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    def train(epoch, model, optimizer, dataloader, criterion, scheduler=None, train=False):
        epoch_loss = 0
        total_items = 0
        correct = np.zeros(len(num_classes))

        if train:
            model.train()
        else:
            model.eval()

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

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
                    correct[i] += get_correct(output[i], y)

            # TODO : label weights
            batch_loss = (label_weights*label_loss).sum()
            epoch_loss += batch_loss.item()
            total_items += data.size(0)

            # print("label_loss :", label_loss)
            # print("batch_loss :", batch_loss)

            # Backwards pass, update learning rates
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

        # END BATCH LOOP

        # Epoch metrics
        unit_loss = epoch_loss / len(dataloader)
        acc = 100 * correct / total_items
        # print("correct post:", correct)
        print("acc post :", acc)
        acc = acc.mean()
        print("EPOCH {}. train={}. Accuracy={:.2f}%. Loss={:.4f}".format(epoch, train, acc, unit_loss))

        if train:
            train_loss.append(unit_loss)
            train_accuracy.append(acc)
        else:
            valid_loss.append(unit_loss)
            valid_accuracy.append(acc)

        
        # TODO : save model

        # END TRAIN FUNCTION

    def time_to_string(t):
        if t > 3600: return "{:.2f} hours".format(t/3600)
        if t > 60: return "{:.2f} minutes".format(t/60)
        else: return "{:.2f} seconds".format(t)

    for epoch in range(num_epochs):
        epoch += 1
        t0 = time.time()
        train(epoch, model, optimizer, train_loader, criterion, scheduler, train=True)
        if validate:
            train(epoch, model, optimizer, test_loader, criterion, scheduler, train=False)
        duration = time.time()-t0
        print("EPOCH {}/{}. Duration={}. Remaing={}".format(epoch, num_epochs, time_to_string(duration), time_to_string(duration*(num_epochs-epoch))))

    # exit()
    # Display loss graphs
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epochs', fontsize=16)
    ax1.set_ylabel('accuracy %', fontsize=16, color=color)
    ax1.plot(range(1, len(train_accuracy)+1), train_accuracy, 'r-', label='training accuracy')
    ax1.plot(range(1, len(valid_accuracy)+1), valid_accuracy, 'y-', label='validation accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('loss', fontsize=16, color=color)  # we already handled the x-label with ax1
    ax2.plot(range(1, len(train_loss)+1), train_loss, 'b-', label='training loss')
    ax2.plot(range(1, len(valid_loss)+1), valid_loss, 'g-', label='validation loss')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()

    plt.show()
