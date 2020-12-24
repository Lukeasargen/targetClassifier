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
from model import BuildMultiTaskTargetNet

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


def get_correct_multi_class(output, target):
    pred = torch.max(output, 1)[1]  # second tensor is indicies
    correct = (pred == target).sum().item()
    return correct

def get_correct_scalar(output, target, *args):
    output = output.squeeze()
    return angle_correct(output, target, *args)

def angle_correct(output, target, angle_threshold):
    return (torch.abs(output-target) < angle_threshold).sum().item()

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
    backbone_features = 512
    avgpool_size = 4  # input_size/16
    filters = [64, 64, 128, 256, 512]
    blocks = [2, 2, 2, 2]
    bottleneck = True
    groups = 1
    width_per_group = None
    dropout_conv = 0.0
    # num classes is defined after dataset is made
    # Training Hyperparameters
    num_epochs = 500
    validate = False
    batch_size = 128
    train_size = 4096
    test_size = 512
    num_workers = 8
    shuffle = False
    drop_last = True
    base_lr = 1e-2
    momentum = 0.9
    weight_decay = 4e-5
    lr_milestones = [150, 440, 480]
    lr_gamma = 0.1
    angle_threshold = 1/180
    # Dataset config
    target_size = 60
    scale = (0.5, 1.0)
    rotation = True

    set_mean = [0.536, 0.429, 0.387]
    set_std = [0.183, 0.176, 0.199]

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
    model = BuildMultiTaskTargetNet(in_channels, backbone_features, avgpool_size, num_classes, filters, blocks,
                bottleneck, groups, dropout_conv).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=base_lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Training
    label_weights = torch.FloatTensor(label_weights).to(device)
    train_loss = np.zeros((num_epochs, len(num_classes)))
    train_accuracy = np.zeros((num_epochs, len(num_classes)))
    valid_loss = np.zeros((num_epochs, len(num_classes)))
    valid_accuracy = np.zeros((num_epochs, len(num_classes)))

    def train(epoch, model, optimizer, dataloader, criterion, train=False):
        epoch_loss = 0
        label_loss_epoch = 0
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
                    correct[i] += get_correct_scalar(output[i], y, angle_threshold)
                else:  # class cross entropy loss
                    # TODO : class weights
                    label_loss[i] = criterion(output[i], y)
                    correct[i] += get_correct_multi_class(output[i], y)

            batch_loss_weighted = (label_weights*label_loss).sum()  # for backprop

            label_loss_epoch += label_loss  # for stats
            epoch_loss += batch_loss_weighted.item()  # for stats
            total_items += data.size(0)  # for stats

            # print("label_loss :", label_loss)
            # print("batch_loss_weighted :", batch_loss_weighted)

            # Backwards pass, update learning rates
            if train:
                optimizer.zero_grad()
                batch_loss_weighted.backward()
                optimizer.step()

        # END BATCH LOOP

        # Epoch metrics
        unit_loss = epoch_loss / len(dataloader)
        acc = 100 * correct / total_items

        print("label_loss_epoch :", label_loss_epoch.tolist())
        print("acc post :", acc)
        print("EPOCH {}. train={}. Accuracy={:.2f}%. Loss={:.4f}".format(epoch, train, acc.mean(), unit_loss))

        if train:
            train_loss[epoch-1] = (label_loss_epoch.cpu().detach().numpy())
            train_accuracy[epoch-1] = (acc)
        else:
            valid_loss[epoch-1] = (label_loss_epoch.cpu().detach().numpy())
            valid_accuracy[epoch-1] = (acc)

        
        # TODO : save model

        # END TRAIN FUNCTION

    def time_to_string(t):
        if t > 3600: return "{:.2f} hours".format(t/3600)
        if t > 60: return "{:.2f} minutes".format(t/60)
        else: return "{:.2f} seconds".format(t)


    t0 = time.time()
    for epoch in range(num_epochs):
        epoch += 1
        t1 = time.time()
        train(epoch, model, optimizer, train_loader, criterion, train=True)
        if validate:
            train(epoch, model, optimizer, test_loader, criterion, train=False)
        duration = time.time()-t1
        print("EPOCH {}/{}. Epoch Duration={}. Run Duration={}. Remaining={}".format(epoch, num_epochs, time_to_string(duration), time_to_string(time.time()-t0), time_to_string(duration*(num_epochs-epoch))))
        
        if scheduler:  # Use pytorch scheduler
            scheduler.step()
        else:  # Janky loss scheduler
            if epoch in lr_milestones:
                new_lr = lr_gamma * optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = new_lr
                print("MILESTONE {}: lr reduced to {}".format(epoch, new_lr))


    # exit()

    # Display loss graphs
    line_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        # 'r-', 'y-', 'b-', 'g-'
    ]


    fig, ax = plt.subplots(2, 1)

    color = 'tab:red'
    ax[0].set_xlabel('epochs', fontsize=16)
    ax[0].set_ylabel('accuracy %', fontsize=16, color=color)
    ax[0].tick_params(axis='y', labelcolor=color)
    for i in range(len(num_classes)):
        ax[0].plot(range(1, num_epochs+1), train_accuracy[:,i], color=line_colors[i], label='label={} train acc'.format(i))
        if validate:
            ax[0].plot(range(1, num_epochs+1), valid_accuracy[:,i], color=line_colors[i], label='label={} val acc'.format(i))
    plt.legend()

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax[1].set_ylabel('loss', fontsize=16, color=color)  # we already handled the x-label with ax1
    ax[1].tick_params(axis='y', labelcolor=color)
    for i in range(len(num_classes)):
        ax[1].plot(range(1, num_epochs+1), train_loss[:,i], color=line_colors[i], label='label={} train loss'.format(i))
        if validate:
            ax[1].plot(range(1, num_epochs+1), valid_loss[:,i], color=line_colors[i], label='label={} val loss'.format(i))
    plt.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
