import copy
import json
import os  # Get files
import time  # Time training

import matplotlib.pyplot as plt
import numpy as np  # Random selection
from PIL import Image
import torch  # Tensor library
import torch.nn as nn  # loss functinos
import torch.optim as optim  # Optimization and schedulers
from torch.utils.data import DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
from torch.autograd import Variable
from torch.cuda import amp

from custom_transforms import CustomTransformation
from dataset_classify import LiveClassifyDataset, FolderClassifyDataset
from logger import Logger
from models.multitask import BuildMultiTaskResnet, save_multitask_resnet
from metrics import WeightedTaskLoss


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def multi_class_loss(output, target):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(output, target)


def get_correct_multi_class(output, target):
    pred = torch.max(output, 1)[1]  # second tensor is indicies
    correct = (pred == target).sum().item()
    return correct

def get_correct_scalar(output, target, *args):
    """ Called in the train loop """
    output = output.squeeze()
    return scalar_correct(output, target, *args)

def scalar_correct(output, target, scalar_threshold):
    return (torch.abs(output-target) < scalar_threshold).sum().item()

def scalar_loss(output, target):
    """ Loss function when the output is a scalar (not a class distribution). """
    loss_func = nn.L1Loss()
    # loss_func = nn.MSELoss()
    return loss_func(output.squeeze(), target)


if __name__ == "__main__" and '__file__' in globals():
    MANUAL_SEED = 103
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)
    # Model Config
    input_size = 32
    in_channels = 3
    backbone_features = 128
    avgpool_size = None
    filters = [32, 32, 64, 128]
    blocks = [2, 2, 2]
    bottleneck = False
    groups = 1
    width_per_group = None
    dropout = 0.0
    # num_classes is defined by the dataset
    # Training Hyperparameters
    num_epochs = 72
    train_size = 16384
    batch_size = 256
    validate = False
    val_size = 2048
    val_batch_size = 256
    num_workers = 4
    shuffle = True
    drop_last = True  # Does multiprocessing handle this well?
    base_lr = 6e-2
    momentum = 0.9
    nesterov = True
    weight_decay = 5e-4  # 0, 1e-5, 3e-5, *1e-4, 3e-4, *5e-4, 3e-4, 1e-3, 1e-2
    use_safe_weight_decay = True   # none on batchnorm or bias terms, I only see 0.5% lower error, could just be noise
    use_fp16 = False
    use_amp = False
    lr_milestones = [62, 68]  # for StepLR
    lr_gamma = 0.1  # for StepLR
    use_exp_lr = False
    lr_exp_gamma = 0.98
    use_lr_ramp_up = True
    lr_ramp_base = 1e-2
    lr_ramp_steps = 23
    use_weighted_loss = True
    sigma_lr = base_lr/1000  # lr of the output variance
    show_graph = True  # use plt to graph, acc and loss
    scalar_threshold = 45/360  # scalar magnitude of difference to be correct
    # Dataset config
    tasks_names = [
        "orientation",
        "shape",
        "letter",
        "shape_color",
        "letter_color",
    ]
    dataset_folder = None # 'images/classify1'  # root directory that has images and labels.csv, if None targets are made during the training
    val_split = 0.03  # percentage of dataset used for validation
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 31
    scale = (0.8, 1.0)
    rotation = True
    expansion_factor = 2  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
    ])

    # If a folder is given, the statistics will be loaded from there
    set_mean = [0.484, 0.522, 0.290]
    set_std = [0.172, 0.156, 0.170]

    save_path = lambda r, t : 'runs/multitask/run{:05d}_{}.pth'.format(r, t)
    if not os.path.exists('runs/multitask'):
        os.makedirs('runs/multitask')

    log_headers = [
        "epoch", "iterations",
        "train_loss", "train_loss_tasks", "train_acc_mean", "train_acc_tasks",
    ]
    if validate:
        log_headers += ["val_loss", "val_loss_tasks", "val_acc_mean", "val_acc_tasks"]

    if use_weighted_loss:
        log_headers += ["sigma"]

    train_transforms = T.Compose([
        CustomTransformation(),
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
    ])

    if dataset_folder:
        full_set = FolderClassifyDataset(folder=dataset_folder, tasks_names=tasks_names, transforms=train_transforms)
        val_size = int(len(full_set)*val_split)
        if validate:
            if val_size < batch_size:
                val_batch_size = val_size
                print("Validate set is smaller than batch size. val_batch_size is reduced to", val_batch_size)
            print("Dataset Split. {}% for validation.".format(100*val_split))
            print("train_size :", len(full_set)-val_size)
            print("val_size :", val_size)
            train_dataset, val_dataset = torch.utils.data.random_split(full_set, [len(full_set)-val_size, val_size])
        else:
            train_dataset = full_set
        set_info = json.load(open(dataset_folder+"/set_info.json"))
        print("Loaded mean and std from folder.")
        set_mean = set_info["mean"]
        set_std = set_info["std"]
        input_size = set_info["input_size"]
        in_channels = set_info["in_channels"]
    else:
        train_dataset = LiveClassifyDataset(length=train_size, tasks_names=tasks_names,
            input_size=input_size, target_size=target_size,
            expansion_factor=expansion_factor, scale=scale, rotation=rotation, bkg_path=bkg_path,
            target_transforms=target_transforms, transforms=train_transforms)
        if validate:
            val_dataset = LiveClassifyDataset(length=val_size, tasks_names=tasks_names,
                input_size=input_size, target_size=target_size,
                expansion_factor=expansion_factor, scale=scale, rotation=rotation, bkg_path=bkg_path,
                target_transforms=target_transforms, transforms=val_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))
    if validate:
        val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size,
                num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))

    if dataset_folder:
        all_classes = set_info["num_classes"]
    else:
        all_classes = train_dataset.gen.num_classes
    print("all_classes :", all_classes)
    num_classes = [all_classes.get(key) for key in tasks_names]
    print("num_classes :", num_classes)

    print("mean :", set_mean)
    print("std :", set_std)

    # Check for cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device :', device)
    model = BuildMultiTaskResnet(backbone_features, num_classes, in_channels, avgpool_size,
                filters, blocks, bottleneck, groups, width_per_group)
    model.set_normalization(mean=set_mean, std=set_std)
    model = model.to(device)


    """ https://github.com/pytorch/pytorch/issues/520#issuecomment-277741834
    training fp16 batchnorm only leads to a 0.5% decrease in accuracy
    1.5x to 4x faster training still looks pretty good
    possibly fine tune after intial training
    """
    def safe_fp16(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.float()
        for child in module.children():
            safe_fp16(child)
        return module

    if use_fp16 and not use_amp:
        print("half model")
        model = model.half()
        model = safe_fp16(model)

    weighted_loss_citerion = None

    """
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    """
    def add_weight_decay(module, weight_decay):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

    if use_safe_weight_decay and weight_decay != 0:
        params = add_weight_decay(model, weight_decay)
    else:
        params = [{'params': model.parameters(), 'lr': base_lr}]

    if use_weighted_loss and len(tasks_names)>1:
        print("Using uncertainty weighted loss.")
        weighted_loss_citerion = WeightedTaskLoss(num_tasks=len(num_classes)).to(device)
        params.append({'params': weighted_loss_citerion.parameters(), 'lr': sigma_lr, 'weight_decay': 0.0})

    optimizer = optim.SGD(params,
                        lr=base_lr,
                        momentum=momentum,
                        nesterov=nesterov,
                        weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    if use_exp_lr:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_exp_gamma)

    scalar = amp.GradScaler()

    if use_lr_ramp_up:
        print("LR RAMP from {} to {} with {} steps.".format(lr_ramp_base, base_lr, lr_ramp_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr_ramp_base

    # Training
    iterations = 0
    train_loss = np.zeros((num_epochs, len(num_classes)))
    train_accuracy = np.zeros((num_epochs, len(num_classes)))
    valid_loss = np.zeros((num_epochs, len(num_classes)))
    valid_accuracy = np.zeros((num_epochs, len(num_classes)))

    sigma = np.zeros((num_epochs, len(num_classes)))
    current_lr = np.zeros((num_epochs, 1))

    def calc_loss(output, target):
        # Computing the loss and training metrics
        batch_correct = np.zeros(len(num_classes))
        tasks_loss = Variable(torch.FloatTensor(len(num_classes))).zero_().to(device)  # TODO : does this need to be on GPU
        log_preds = np.zeros((batch_size, len(num_classes)))  # track predictions for logging
        for i in range(len(num_classes)):
            y = target
            if target.ndim != 1:  # check for multiple task target
                y = target[:, i]
            if output[i].shape[1] == 1:  # output is a scalar not class logits
                tasks_loss[i] = scalar_loss(output[i], y)
                batch_correct[i] += get_correct_scalar(output[i], y, scalar_threshold)
                log_preds[:, i] = output[i].detach().cpu().squeeze().numpy()                   
            else:  # class cross entropy loss
                # TODO : class weights
                y = y.long()  # Requires dtype = long
                tasks_loss[i] = multi_class_loss(output[i], y)
                batch_correct[i] += get_correct_multi_class(output[i], y)
                preds_percent, class_idx = torch.max(output[i], dim=1)
                log_preds[:, i] = class_idx.detach().cpu().squeeze().numpy() 

        if weighted_loss_citerion:
            batch_loss = weighted_loss_citerion(output, target)
        else:
            batch_loss = tasks_loss.sum()  # for back propagation

        return batch_loss, tasks_loss, batch_correct, log_preds

    def train(epoch, model, optimizer, dataloader, scalar, train=False):
        global iterations
        epoch_loss = 0
        tasks_loss_epoch = 0
        total_items = 0
        correct = np.zeros(len(num_classes))

        if train:
            model.train()
        else:
            model.eval()  # lock batchnorm layers

        c = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            c +=1
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            if use_amp:
                with amp.autocast():
                    output = model(data, dropout)
                    batch_loss, tasks_loss, batch_correct, log_preds = calc_loss(output, target)
            else:
                if use_fp16:
                    data = data.half()
                if train:
                    output = model(data, dropout)
                else:
                    with torch.no_grad():
                        output = model(data, dropout)
                batch_loss, tasks_loss, batch_correct, log_preds = calc_loss(output, target)

            correct += batch_correct
            tasks_loss_epoch += tasks_loss  # for stats
            epoch_loss += batch_loss.item()  # for stats
            total_items += data.size(0)  # for stats

            # Backwards pass
            if train:
                if use_amp:
                    with amp.autocast():
                        scalar.scale(batch_loss).backward()
                        scalar.step(optimizer)
                        scalar.update()
                else:
                    batch_loss.backward()
                    optimizer.step()
                iterations += 1

        # END BATCH LOOP

        # Epoch metrics
        unit_loss = epoch_loss / len(dataloader)
        acc_tasks = 100 * correct / total_items
        loss_tasks = (tasks_loss_epoch.detach().cpu().numpy()/c)

        if train:
            train_loss[epoch-1] = loss_tasks
            train_accuracy[epoch-1] = (acc_tasks)
            if weighted_loss_citerion:
                sigma[epoch-1] = weighted_loss_citerion.sigma.data.detach().cpu().numpy()
                print("sigma :", sigma[epoch-1])
            current_lr[epoch-1] = optimizer.param_groups[0]['lr']
        else:
            valid_loss[epoch-1] = loss_tasks
            valid_accuracy[epoch-1] = (acc_tasks)

        # print("tasks_loss_epoch :", tasks_loss_epoch.tolist())
        print("acc :", acc_tasks)
        print("err :", 100-acc_tasks)
        # print("sigma :", weighted_loss_citerion.sigma.data.detach().cpu().numpy())
        print("EPOCH {}. train={}. Accuracy={:.2f}%. Loss={:.4f}".format(epoch, train, acc_tasks.mean(), unit_loss))

        if train:
            prefix = "train"
        else:
            prefix = "val"
        metrics = {
            f"{prefix}_loss": unit_loss,
            f"{prefix}_loss_tasks": loss_tasks,
            f"{prefix}_acc_mean": acc_tasks.mean(),
            f"{prefix}_acc_tasks": acc_tasks, 
        }
        return metrics

        # END TRAIN FUNCTION

    def time_to_string(t):
        if t > 3600: return "{:.2f} hours".format(t/3600)
        if t > 60: return "{:.2f} minutes".format(t/60)
        else: return "{:.2f} seconds".format(t)

    current_run = 0
    with open('runs/LASTRUN.txt') as f:
        current_run = int(f.read()) + 1
    with open('runs/LASTRUN.txt', 'w') as f:
        f.write("%s\n" % current_run)


    # Create logger
    log_name = "run{:05d}".format(current_run)

    logger = Logger(name=log_name, headers=log_headers, folder="runs/multitask")

    best_loss = None
    best_loss_epoch = 0

    t0 = time.time()
    for epoch in range(num_epochs):
        epoch += 1
        t1 = time.time()
        epoch_metrics = {"epoch": epoch}
        train_metrics = train(epoch, model, optimizer, train_loader, scalar, train=True)
        epoch_metrics.update(train_metrics)
        if validate:
            val_metrics = train(epoch, model, optimizer, val_loader, scalar, train=False)
            epoch_metrics.update(val_metrics)
        epoch_metrics.update({"iterations": iterations})

        duration = time.time()-t1
        print("EPOCH {}/{}. Epoch Duration={}. Run Duration={}. Remaining={}".format(epoch, num_epochs, time_to_string(duration), time_to_string(time.time()-t0), time_to_string(duration*(num_epochs-epoch))))
        
        if use_lr_ramp_up and (epoch-1) < lr_ramp_steps:
            new_lr = lr_ramp_base + (epoch)*((base_lr-lr_ramp_base)/lr_ramp_steps)
            print("LR RAMP UP. Step {}. lr set to {:.6f}".format(epoch, new_lr))
            for group in optimizer.param_groups:
                group["lr"] = new_lr

        if scheduler:  # Use pytorch scheduler
            scheduler.step()

        which_loss = "val_loss" if validate else "train_loss"
        if best_loss == None:
            best_loss = epoch_metrics[which_loss]
            best_model = copy.deepcopy(model)
        elif epoch_metrics[which_loss] < best_loss:
            best_loss_epoch = epoch
            best_loss = epoch_metrics[which_loss]
            best_model = copy.deepcopy(model)

        if use_weighted_loss:
            epoch_metrics.update({"sigma" : sigma[epoch-1]})

        logger.update(epoch_metrics)

        # END TRAIN LOOP

    print("Best Loss Epoch {} : {:.4f}".format(best_loss_epoch, best_loss))

    save_multitask_resnet(best_model, save_path(current_run, "best_loss"), input_size)
    save_multitask_resnet(model, save_path(current_run, "final"), input_size)

    if show_graph:

        # Display loss graphs
        line_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        fig, ax = plt.subplots(2, 2, figsize=(12,8))  # w, h

        color = 'tab:red'
        ax[0][0].set_xlabel('epochs', fontsize=16)
        ax[0][0].set_ylabel('accuracy %', fontsize=16, color=color)
        ax[0][0].tick_params(axis='y', labelcolor=color)
        for i in range(len(num_classes)):
            ax[0][0].plot(range(1, num_epochs+1), train_accuracy[:,i], color=line_colors[i], label=tasks_names[i])
            if validate:
                ax[0][0].plot(range(1, num_epochs+1), valid_accuracy[:,i], color=line_colors[i], linestyle='dashed')
        ax[0][0].legend()

        color = 'tab:blue'
        ax[0][1].set_ylabel('loss', fontsize=16, color=color)  # we already handled the x-label with ax1
        ax[0][1].tick_params(axis='y', labelcolor=color)
        for i in range(len(num_classes)):
            ax[0][1].plot(range(1, num_epochs+1), train_loss[:,i], color=line_colors[i], label=tasks_names[i])
            if validate:
                ax[0][1].plot(range(1, num_epochs+1), valid_loss[:,i], color=line_colors[i], linestyle='dashed')

        ax[1][0].set_ylabel('learning rate', fontsize=16)  # we already handled the x-label with ax1
        ax[1][0].tick_params(axis='y')
        ax[1][0].plot(range(1, num_epochs+1), current_lr)

        if weighted_loss_citerion:
            color = 'tab:green'
            ax[1][1].set_ylabel('sigma', fontsize=16, color=color)  # we already handled the x-label with ax1
            ax[1][1].tick_params(axis='y', labelcolor=color)
            for i in range(len(num_classes)):
                ax[1][1].plot(range(1, num_epochs+1), sigma[:,i], color=line_colors[i], label=tasks_names[i])


        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        fig.savefig("runs/multitask/run{:05d}_metrics.png".format(current_run), bbox_inches='tight')

        plt.show()
