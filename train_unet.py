
import os
import time
from typing import List

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np  # Random selection
from PIL import Image
import torch  # Tensor library
import torch.nn as nn  # loss functinos
import torch.optim as optim  # Optimization and schedulers
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
from torch.cuda import amp

from custom_transforms import CustomTransformation, AddGaussianNoise
from dataset_segment import LiveSegmentDataset
from logger import Logger
from metrics import pixel_accuracy, jaccard_iou, dice_coeff, tversky_measure, focal_metric

from models.unet import UNet, save_unet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def view_outputs(model, train_dataset, threshold=0.5):
    nrows = 5
    ncols = 2
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = train_dataset.gen.gen_segment(fill_prob=0.4)
            x = val_transforms(img).to(device).unsqueeze(0)
            out = model.predict(x).detach().cpu().numpy()
            preds = (np.repeat(out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            row.append(img)
            fp_mask = ((1-np.array(mask)/255) * preds*255)
            fn_mask = (np.array(mask) * (1-preds))
            row.append(preds*255)  # multiple to 255 rgb scale
            row.append(fp_mask)
            row.append(fn_mask)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    # print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/unet_train_demo.png")


def calc_loss(logits: torch.Tensor, targets: torch.Tensor):
    # bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    # focal_loss = focal_metric(logits, targets, alpha=0.5, gamma=2.0)

    preds = torch.sigmoid(logits)
    jaccard_loss = 1.0 - jaccard_iou(preds, targets, smooth=1.0)
    # dice_loss = 1.0 - dice_coeff(preds, targets, smooth=1.0)
    # tversky_loss = 1.0 - tversky_measure(preds, targets, alpha=0.3, beta=0.7, smooth=1.0)

    return jaccard_loss


def calc_metrics(logits: torch.Tensor, targets: torch.Tensor):
    preds = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    metrics = {
        "acc": pixel_accuracy(preds, targets, threshold=0.5).item(),
        "bce": bce.item(),
        "jaccard": jaccard_iou(preds, targets, smooth=1.0).item(),
        "dice": dice_coeff(preds, targets, smooth=1.0).item(),
        "tversky": tversky_measure(preds, targets, alpha=0.3, beta=0.7, smooth=1.0).item(),
        "focal": focal_metric(logits, targets, alpha=0.5, gamma=2.0).item()
    }
    return metrics


if __name__ == "__main__":
    MANUAL_SEED = 42
    set_seed(MANUAL_SEED)
    torch.backends.cudnn.benchmark = False

    # Training data
    save_final = True
    run_validation = True
    graph_metrics = True
    view_results = True
    view_threshold = 0.5

    # Model Config
    in_channels = 3
    out_channels = 1
    model_type = "unet"  # unet, unet_nested, unet_nested_deep
    filters = 4  # 16
    activation = "relu"  # relu, leaky_relu, silu, mish

    # Training Hyperparameters
    input_size = 192 # 400
    train_epochs = 280 # 20
    val_epochs = 4
    train_size = 256 # 8000
    batch_size = 16 # 4
    shuffle = False
    num_workers = 6
    drop_last = False

    # Mixed precision
    use_amp = False
    detect_grad_failures = False  # sets autograd.detect_anomaly

    # Optimization
    optim_type = 'adamw'  # sgd 1e-1, rmsprop 1e-3, adam 4e-3, adamw 4e-3, adagrad 1e-2
    base_lr = 4e-3  # 1e-1
    momentum = 0.9  # 0.9
    nesterov = True
    weight_decay = 5e-4  # 0, 1e-5, 3e-5, *1e-4, 3e-4, *5e-4, 3e-4, 1e-3, 1e-2
    clip_grad_max = None  # None is no clipping, otherwise use a positive float
    scheduler_type = 'step'  # step, plateau, exp
    lr_milestones = [200, 250]  # for StepLR, [10, 15]
    lr_gamma = 0.2
    plateau_patience = 20

    # Dataset parameters
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 20  # Smallest target size
    fill_prob = 0.9
    expansion_factor = 1  # generate higher resolution targets and downscale, improves aliasing effects
    set_mean = [0.5, 0.5, 0.5]
    set_std = [0.5, 0.5, 0.5]
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
    ])
    train_transforms = T.Compose([
        CustomTransformation(),
        T.ToTensor(),
        AddGaussianNoise(std=0.01)
    ])
    val_transforms = T.Compose([
        T.ToTensor(),
    ])

    # Prepare datasets
    train_dataset = LiveSegmentDataset(length=train_size,
        input_size=input_size, target_size=target_size,
        expansion_factor=expansion_factor, bkg_path=bkg_path, fill_prob=fill_prob,
        target_transforms=target_transforms, transforms=train_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))
    
    print("mean :", set_mean)
    print("std :", set_std)

    # Create model and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device :', device)

    model = UNet(in_channels, out_channels, model_type, filters, activation, set_mean, set_std).to(device)

    # Uncomment to test the model before training
    # view_outputs(model, train_dataset, threshold=0.5); exit()

    # TODO : split params so there is no weight decay on batchnorm


    # Setup optimizer
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=momentum)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optim_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Setup scheduler
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=plateau_patience, verbose=True)
    elif scheduler_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        scheduler = None

    # Setup amp
    if use_amp:
        scalar = amp.GradScaler()

    # Setup run data
    base_save = f"runs/unet"
    save_path = lambda r, t : base_save + '/run{:05d}_{}.pth'.format(r, t)
    if not os.path.exists(base_save):
        os.makedirs(base_save)
    
    def time_to_string(t):
        if t > 3600: return "{:.2f} hours".format(t/3600)
        if t > 60: return "{:.2f} minutes".format(t/60)
        else: return "{:.2f} seconds".format(t)

    current_run = 0
    with open('runs/LASTRUN.txt') as f:
        current_run = int(f.read()) + 1
    with open('runs/LASTRUN.txt', 'w') as f:
        f.write("%s\n" % current_run)
    print("current run :", current_run)

    # Create logger
    log_name = "run{:05d}".format(current_run)
    log_headers = ['epoch', 'iterations', 'elapsed_time', 'lr', 'train_loss',
                    'acc', 'bce', 'jaccard', 'dice', 'tversky', 'focal']
    logger = Logger(name=log_name, headers=log_headers, folder=base_save)

    run_stats = []

    # Training loop
    print(" * Start training...")
    t0 = time.time()
    iterations = 0
    for epoch in range(train_epochs):  # loop over the dataset multiple times

        t1 = time.time()
        epoch_loss_total = 0.0
        batch_metrics_total = Counter({})
        model.train()
        model.freeze_norm()
        for batch_idx, (data, true_masks) in enumerate(train_loader):
            data = data.to(device=device)
            true_masks = true_masks.to(device=device)

            optimizer.zero_grad()  # TODO : test each optimizer for set_to_none=True

            if use_amp:
                with amp.autocast():
                    with torch.autograd.set_detect_anomaly(detect_grad_failures):
                        logits = model(data)
                    loss = calc_loss(logits, true_masks)
                scalar.scale(loss).backward()
                if clip_grad_max != None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)
                scalar.step(optimizer)
                scalar.update()
            else:
                logits = model(data)
                loss = calc_loss(logits, true_masks)
                loss.backward()
                if clip_grad_max != None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)
                optimizer.step()

            iterations += 1

            # Update running metrics           
            batch_metrics = calc_metrics(logits, true_masks)
            epoch_loss_total += loss.item()
            batch_metrics_total += Counter(batch_metrics)

        # END EPOCH LOOP

        epoch_loss = epoch_loss_total/len(train_loader)
        batch_metrics_avg = {k: v / len(train_loader) for k, v in batch_metrics_total.items()}
        epoch_metrics = {
            "epoch": epoch+1,
            "iterations": iterations,
            "elapsed_time": time.time()-t0,
            "lr": optimizer.param_groups[0]['lr'],
            "train_loss": epoch_loss
        }
        epoch_metrics.update(batch_metrics_avg)
        logger.update(epoch_metrics)
        run_stats.append(epoch_metrics)

        if scheduler:
            if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(epoch_loss)
            elif type(scheduler) == optim.lr_scheduler.MultiStepLR:
                scheduler.step()

        duration = time.time()-t1
        remaining = duration*(train_epochs-epoch-1)
        print("epoch {}. {} iterations: {}. loss={:.04f}. lr={:.06f}. elapsed={}. remaining={}.".format(epoch+1, iterations, time_to_string(duration), epoch_loss, optimizer.param_groups[0]['lr'], time_to_string(time.time()-t0), time_to_string(remaining)))

    # END TRAIN LOOP
    print('Finished Training. Duration={}. {} iterations.'.format(time_to_string(time.time()-t0), iterations))
    print("Final stats:")
    print("loss={:.05f}. acc={:.04f}. bce={:.05f}. jaccard={:.04f}. dice={:.04f}. tversky={:.04f}. focal={:.06f}.".format(epoch_metrics["train_loss"], epoch_metrics["acc"], epoch_metrics["bce"], epoch_metrics["jaccard"], epoch_metrics["dice"], epoch_metrics["tversky"], epoch_metrics["focal"]))
    

    # Save Model
    if save_final:
        save_unet(model, save_path(current_run, "final"))


    if run_validation:
        print(" * Running validation...")
        set_seed(MANUAL_SEED)
        model.eval()
        val_loss_total = 0.0
        val_metrics_total = Counter({})
        t0 = time.time()
        for epoch in range(val_epochs):
            t1 = time.time()
            for batch_idx, (data, true_masks) in enumerate(train_loader):
                data = data.to(device=device)
                true_masks = true_masks.to(device=device)
                with torch.no_grad():
                    if use_amp:
                        with amp.autocast():
                            logits = model(data)
                        loss = calc_loss(logits, true_masks)
                    else:
                        logits = model(data)
                        loss = calc_loss(logits, true_masks)
                batch_metrics = calc_metrics(logits, true_masks)
                val_loss_total += loss.item()
                val_metrics_total += Counter(batch_metrics)
            duration = time.time()-t1
            print("validation {}/{}. elapsed={}. remaining={}.".format(epoch+1, val_epochs, time_to_string(time.time()-t0), time_to_string(remaining)))
        # END VALIDATION LOOP
        val_loss = val_loss_total/(val_epochs*len(train_loader))
        val_metrics = {k: v/(val_epochs*len(train_loader)) for k, v in val_metrics_total.items()}
        print("Validation stats:")
        print("loss={:.05f}. acc={:.04f}. bce={:.05f}. jaccard={:.04f}. dice={:.04f}. tversky={:.04f}. focal={:.06f}.".format(val_loss, val_metrics["acc"], val_metrics["bce"], val_metrics["jaccard"], val_metrics["dice"], val_metrics["tversky"], val_metrics["focal"]))


    if view_results:
        view_outputs(model, train_dataset, threshold=view_threshold)
    

    # Graph metrics
    if graph_metrics:
        line_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        fig, ax = plt.subplots(2, 3, figsize=(12,8))  # w, h
        font_size = 16

        # loss, top 0
        ax[0][0].set_xlabel('epochs', fontsize=font_size)
        ax[0][0].set_ylabel('train loss', fontsize=font_size)
        ax[0][0].set_yscale('log')
        ax[0][0].tick_params(axis='y')
        ax[0][0].plot(range(1, train_epochs+1), [x["train_loss"] for x in run_stats], color=line_colors[0], label="train loss")

        # bce, top 1
        ax[0][1].set_xlabel('epochs', fontsize=font_size)
        ax[0][1].set_ylabel('bce', fontsize=font_size)
        ax[0][1].set_yscale('log')
        ax[0][1].tick_params(axis='y')
        ax[0][1].plot(range(1, train_epochs+1), [x["bce"] for x in run_stats], color=line_colors[0], label="bce loss")

        # focal, top 2
        ax[0][2].set_xlabel('epochs', fontsize=font_size)
        ax[0][2].set_ylabel('focal', fontsize=font_size)
        ax[0][2].set_yscale('log')
        ax[0][2].tick_params(axis='y')
        ax[0][2].plot(range(1, train_epochs+1), [x["focal"] for x in run_stats], color=line_colors[0], label="focal loss")

        # lr, bot 0
        ax[1][0].set_xlabel('epochs', fontsize=font_size)
        ax[1][0].set_ylabel('lr', fontsize=font_size)
        ax[1][0].set_yscale('log')
        ax[1][0].tick_params(axis='y')
        ax[1][0].plot(range(1, train_epochs+1), [x["lr"] for x in run_stats], color=line_colors[0], label="lr")

        # coef, bot 1
        ax[1][1].set_xlabel('epochs', fontsize=font_size)
        ax[1][1].set_ylabel('coefficients', fontsize=font_size)
        ax[1][1].tick_params(axis='y')
        coef = ["jaccard", "dice", "tversky"]
        for i in range(len(coef)):
            ax[1][1].plot(range(1, train_epochs+1), [x[coef[i]] for x in run_stats], color=line_colors[i], label=coef[i])
        ax[1][1].legend()

        # acc, bot 2
        ax[1][2].set_xlabel('epochs', fontsize=font_size)
        ax[1][2].set_ylabel('acc', fontsize=font_size)
        # ax[1][2].set_yscale('log')
        ax[1][2].tick_params(axis='y')
        ax[1][2].plot(range(1, train_epochs+1), [x["acc"] for x in run_stats], color=line_colors[0], label="acc")


        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("{}/run{:05d}_metrics.png".format(base_save, current_run), bbox_inches='tight')
        plt.show()
