
import os
import time

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np  # Random selection
from PIL import Image
import torch  # Tensor library
import torch.nn as nn  # loss functinos
import torch.optim as optim  # Optimization and schedulers
from torch.utils.data import DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing

from custom_transforms import CustomTransformation
from dataset_segment import LiveSegmentDataset
from models.unet import UNet, save_unet
from logger import Logger
from metrics import pixel_accuracy, jaccard_iou, dice_coeff, tversky_measure, focal_metric


def view_outputs(model, train_dataset, threshold=0.5):
    model.eval()
    nrows = 5
    ncols = 2
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = train_dataset.gen.gen_segment(fill_prob=0.3)
            x = val_transforms(img).to(device).unsqueeze(0)
            with torch.no_grad():
                out = model(x).detach().cpu().numpy()
            preds = (np.repeat(out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            row.append(img)
            row.append(mask)
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


def loss_func(preds, targets, metrics_tracked=[]):
    out = {}

    acc = pixel_accuracy(preds, targets, threshold=0.5)
    bce_loss = nn.BCELoss()(preds, targets)
    jaccard = jaccard_iou(preds, targets, smooth=1)
    diceCoeff = dice_coeff(preds, targets, smooth=1)
    tversky = tversky_measure(preds, targets, alpha=0.3, beta=0.7, smooth=1)
    focal_loss = focal_metric(preds, targets, alpha=0.5, gamma=2)

    if 'acc' in metrics_tracked: out.update({"acc": acc.item()})
    if 'bce' in metrics_tracked: out.update({"bce": bce_loss.item()})
    if 'jaccard' in metrics_tracked: out.update({"jaccard": jaccard.item()})
    if 'dice' in metrics_tracked: out.update({"dice": diceCoeff.item()})
    if 'tversky' in metrics_tracked: out.update({"tversky": tversky.item()})
    if 'focal' in metrics_tracked: out.update({"focal": focal_loss.item()})

    jaccard_loss = 1 - jaccard
    dice_loss = 1 - diceCoeff
    tversky_loss = 1 - tversky

    loss = bce_loss + jaccard_loss + tversky_loss
    return loss, out


if __name__ == "__main__":
    MANUAL_SEED = 42
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)

    # Training data
    graph_metrics = True
    view_results = True
    view_threshold = 0.95
    metrics_tracked = [
        'acc',
        'bce',
        'jaccard',
        'dice',
        'tversky',
        'focal'
    ]  # calculated in metrics()

    # Model Config
    input_size = 256 # 400
    in_channels = 3
    out_channels = 1
    filters = 16  # 16

    # Training Hyperparameters
    num_epochs = 26
    train_size = 256 # 8000
    batch_size = 16 # 4
    shuffle = False
    num_workers = 8
    drop_last = True

    # Optimization
    base_lr = 1e-1  # 1e-1
    momentum = 0.9  # 0.9
    nesterov = True
    weight_decay = 5e-4  # 0, 1e-5, 3e-5, *1e-4, 3e-4, *5e-4, 3e-4, 1e-3, 1e-2
    lr_milestones = [16, 22]  # for StepLR, [10, 15]
    lr_gamma = 0.1  # for StepLR, 0.2

    # Dataset parameters
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 20  # Smallest target size
    fill_prob = 0.9
    expansion_factor = 1  # generate higher resolution targets and downscale, improves aliasing effects

    set_mean = [0.4792167,  0.52474296, 0.27591285]
    set_std = [0.17430544, 0.15489028, 0.151296  ]

    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
    ])
    train_transforms = T.Compose([
        CustomTransformation(),
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
    ])
    val_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=set_mean, std=set_std)
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

    model = UNet(in_channels, out_channels, filters).to(device)

    # view_outputs(model, train_dataset, threshold=0.5); exit()

    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Setup run data
    save_path = lambda r, t : 'runs/unet/run{:05d}_{}.pth'.format(r, t)
    if not os.path.exists('runs/unet'):
        os.makedirs('runs/unet')
    
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
    log_headers = ['epoch', 'iterations', 'elapsed_time', 'lr', 'train_loss'] + metrics_tracked
    logger = Logger(name=log_name, headers=log_headers, folder="runs/unet")

    run_stats = []

    # Training loop
    t0 = time.time()
    iterations = 0
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        t1 = time.time()
        epoch_loss_total = 0.0
        batch_metrics_total = Counter({})
        for batch_idx, (data, true_masks) in enumerate(train_loader):
            data = data.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            masks_pred = model(data)
            loss, batch_metrics = loss_func(masks_pred, true_masks, metrics_tracked)
            epoch_loss_total += loss.item()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            iterations += 1

            # Update running metrics
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
        print("epoch_metrics :", epoch_metrics)
        logger.update(epoch_metrics)
        run_stats.append(epoch_metrics)

        if scheduler:
            scheduler.step()

        duration = time.time()-t1
        remaining = duration*(num_epochs-epoch)
        print("epoch {}. loss={:.04f}. lr={:.06f}. Duration={}. remaining={}.".format(epoch+1, epoch_loss, optimizer.param_groups[0]['lr'], time_to_string(duration), time_to_string(remaining)))
        print("epoch_metrics :", epoch_metrics)

    # END TRAIN LOOP

    print('Finished Training. Duration={}.'.format(time_to_string(time.time()-t0)))
    print("Final stats:")
    print("loss={:.05f}. acc={:.05f}. bce={:.05f}. jaccard={:.05f}. dice={:.05f}. tversky={:.05f}. focal={:.05f}.".format(epoch_metrics["train_loss"], epoch_metrics["acc"], epoch_metrics["bce"], epoch_metrics["jaccard"], epoch_metrics["dice"], epoch_metrics["tversky"], epoch_metrics["focal"]))
    
    # Save Model
    save_unet(model, save_path(current_run, "final"), set_mean, set_std)


    if view_results:
        view_outputs(model, train_dataset, threshold=view_threshold)
    

    # Graph metrics
    if graph_metrics:
        line_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        fig, ax = plt.subplots(2, 2)
        font_size = 16

        # Train Loss. Top right
        ax[0][0].set_xlabel('epochs', fontsize=font_size)
        # ax[0][0].set_ylabel('train loss', fontsize=font_size)
        ax[0][0].set_yscale('log')
        ax[0][0].tick_params(axis='y')
        ax[0][0].plot(range(1, num_epochs+1), [x["train_loss"] for x in run_stats], color=line_colors[0], label="train loss")
        ax[0][0].legend()

        # Other coefficients. Top left
        ax[0][1].set_xlabel('epochs', fontsize=font_size)
        # ax[0][1].set_ylabel('coefficients', fontsize=font_size)
        ax[0][1].tick_params(axis='y')
        coef = ["jaccard", "dice", "tversky"]
        for i in range(len(coef)):
            ax[0][1].plot(range(1, num_epochs+1), [x[coef[i]] for x in run_stats], color=line_colors[i], label=coef[i])
        ax[0][1].legend()

        # bce. Bottom right
        ax[1][0].set_xlabel('epochs', fontsize=font_size)
        # ax[1][0].set_ylabel('bce', fontsize=font_size)
        ax[1][0].set_yscale('log')
        ax[1][0].tick_params(axis='y')
        ax[1][0].plot(range(1, num_epochs+1), [x["bce"] for x in run_stats], color=line_colors[0], label="bce loss")
        ax[1][0].legend()

        # focal. Bottom Left
        ax[1][1].set_xlabel('epochs', fontsize=font_size)
        # ax[1][1].set_ylabel('focal', fontsize=font_size)
        ax[1][1].set_yscale('log')
        ax[1][1].tick_params(axis='y')
        ax[1][1].plot(range(1, num_epochs+1), [x["focal"] for x in run_stats], color=line_colors[0], label="focal loss")
        ax[1][1].legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("images/runs/unet/run_{:05d}.png".format(current_run), bbox_inches='tight')
        plt.show()

