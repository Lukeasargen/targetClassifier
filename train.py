import os  # Get files
import csv  # CSV logger
import time  # Time training
import datetime
import numpy as np  # Random selection
import matplotlib.pyplot as plt
import torch  # Tensor library
import torch.nn as nn  # loss functinos
import torch.optim as optim  # Optimization and schedulers
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
from torch.autograd import Variable

from generate_samples import TargetGenerator
from model import BuildMultiTaskTargetNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Logger():
    def __init__(self, name):
        self.name = name + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = None

    def _make_file(self):
        self.path = self.name + ".csv"

    def update(self, epoch, traget, output):
        pass


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
        return self.transforms(x), torch.tensor(list(y.values()), dtype=torch.float).squeeze()


class WeightedTaskLoss(nn.Module):
    """ https://arxiv.org/abs/1703.04977 """
    def __init__(self, num_tasks):
        super(WeightedTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        self.sigma = nn.Parameter(torch.ones(num_tasks))

    def forward(self, tasks_loss):
        # sigma is 1
        # epsilon = 1e-9
        # precision  = (0.5*tasks_loss/(self.sigma+epsilon)**2).sum()
        # loss = precision + torch.log(torch.prod(self.sigma+epsilon))
        loss = (torch.exp(-self.sigma) * tasks_loss + self.sigma).sum()
        return loss

def get_correct_multi_class(output, target):
    pred = torch.max(output, 1)[1]  # second tensor is indicies
    correct = (pred == target).sum().item()
    return correct

def get_correct_scalar(output, target, *args):
    output = output.squeeze()
    return angle_correct(output, target, *args)

def angle_correct(output, target, angle_threshold):
    return (torch.abs(output-target) < angle_threshold).sum().item()

def multi_class_loss(output, target):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(output, target)

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
    backbone_features = 2048
    avgpool_size = 4  # input_size/16
    filters = [64, 256, 512, 1024, 2048]
    blocks = [3, 4, 6, 3]
    bottleneck = True
    groups = 32
    width_per_group = 4
    dropout_conv = 0.0
    # num_classes is defined by the dataset
    # Training Hyperparameters
    num_epochs = 250
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
    lr_milestones = [210, 230, 240]
    lr_gamma = 0.1
    angle_threshold = 10/180
    # Dataset config
    target_size = 60
    scale = (0.5, 1.0)
    rotation = True

    set_mean = [0.538, 0.427, 0.380]
    set_std = [0.183, 0.175, 0.200]

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
    print("num_classes :", num_classes)

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
    weighted_loss_citerion = None #WeightedTaskLoss(num_tasks=len(num_classes)).to(device)
    params = [{'params': model.parameters(), 'lr': base_lr}]
    if weighted_loss_citerion:
        params.append({'params': weighted_loss_citerion.parameters(), 'lr': base_lr/100})
    optimizer = optim.SGD(params,
                        lr=base_lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    # Training
    train_loss = np.zeros((num_epochs, len(num_classes)))
    train_accuracy = np.zeros((num_epochs, len(num_classes)))
    valid_loss = np.zeros((num_epochs, len(num_classes)))
    valid_accuracy = np.zeros((num_epochs, len(num_classes)))
    sigma = np.zeros((num_epochs, len(num_classes)))

    def train(epoch, model, optimizer, dataloader, train=False):
        epoch_loss = 0
        tasks_loss_epoch = 0
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
            tasks_loss = Variable(torch.FloatTensor(len(num_classes))).zero_().to(device)
            log_preds = np.zeros((batch_size, len(num_classes)))  # track predictions for logging
            for i in range(len(num_classes)):
                y = target
                if target.ndim != 1:
                    y = target[:, i]
                if output[i].shape[1] == 1:  # output is a scalar not a class distribution
                    tasks_loss[i] = scalar_loss(output[i], y)
                    correct[i] += get_correct_scalar(output[i], y, angle_threshold)
                    log_preds[:, i] = output[i].detach().cpu().squeeze().numpy()                   
                else:  # class cross entropy loss
                    # TODO : class weights
                    y = y.long()  # Requires dtype = long
                    tasks_loss[i] = multi_class_loss(output[i], y)
                    correct[i] += get_correct_multi_class(output[i], y)
                    preds_percent, class_idx = torch.max(output[i], dim=1)
                    log_preds[:, i] = class_idx.detach().cpu().squeeze().numpy() 
                    # print("preds_percent :", preds_percent)
                    # print("class_idx :", class_idx)

            # print("target :", target)
            # print("log_preds :", log_preds)

            # print("tasks_loss :", tasks_loss)
            if weighted_loss_citerion:
                # print("weighted")
                batch_loss = weighted_loss_citerion(tasks_loss)
            else:
                # print("sum")
                batch_loss = tasks_loss.sum()  # for back propagation

            # print("weighted_loss_citerion :", weighted_loss_citerion(tasks_loss))
            # print("sum :", tasks_loss.sum())
            # print("batch_loss :", batch_loss)


            tasks_loss_epoch += tasks_loss  # for stats
            epoch_loss += batch_loss.item()  # for stats
            total_items += data.size(0)  # for stats

            # print("tasks_loss :", tasks_loss)
            # print("batch_loss :", batch_loss)

            # Backwards pass, update learning rates
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        # END BATCH LOOP

        # Epoch metrics
        unit_loss = epoch_loss / len(dataloader)
        acc = 100 * correct / total_items

        print("tasks_loss_epoch :", tasks_loss_epoch.tolist())
        print("acc post :", acc)
        print("EPOCH {}. train={}. Accuracy={:.2f}%. Loss={:.4f}".format(epoch, train, acc.mean(), unit_loss))

        if train:
            train_loss[epoch-1] = (tasks_loss_epoch.detach().cpu().numpy())
            train_accuracy[epoch-1] = (acc)
            if weighted_loss_citerion:
                sigma[epoch-1] = weighted_loss_citerion.sigma.data.detach().cpu().numpy()
        else:
            valid_loss[epoch-1] = (tasks_loss_epoch.detach().cpu().numpy())
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
        train(epoch, model, optimizer, train_loader, train=True)
        if validate:
            train(epoch, model, optimizer, test_loader, train=False)
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
    labels_str = [
        "Orientation", "Shape", "Letter", "Shape Color", "Letter Color"
    ]

    num_fig = 2
    if weighted_loss_citerion:
        num_fig = 3

    fig, ax = plt.subplots(num_fig, 1)

    color = 'tab:red'
    ax[0].set_xlabel('epochs', fontsize=16)
    ax[0].set_ylabel('accuracy %', fontsize=16, color=color)
    ax[0].tick_params(axis='y', labelcolor=color)
    for i in range(len(num_classes)):
        ax[0].plot(range(1, num_epochs+1), train_accuracy[:,i], color=line_colors[i], label=labels_str[i])
        if validate:
            ax[0].plot(range(1, num_epochs+1), valid_accuracy[:,i], color=line_colors[i], label=labels_str[i])

    color = 'tab:blue'
    ax[1].set_ylabel('loss', fontsize=16, color=color)  # we already handled the x-label with ax1
    ax[1].tick_params(axis='y', labelcolor=color)
    for i in range(len(num_classes)):
        ax[1].plot(range(1, num_epochs+1), train_loss[:,i], color=line_colors[i], label=labels_str[i])
        if validate:
            ax[1].plot(range(1, num_epochs+1), valid_loss[:,i], color=line_colors[i], label=labels_str[i])

    if weighted_loss_citerion:
        color = 'tab:green'
        ax[2].set_ylabel('sigma', fontsize=16, color=color)  # we already handled the x-label with ax1
        ax[2].tick_params(axis='y', labelcolor=color)
        for i in range(len(num_classes)):
            ax[2].plot(range(1, num_epochs+1), sigma[:,i], color=line_colors[i], label=labels_str[i])

    plt.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
