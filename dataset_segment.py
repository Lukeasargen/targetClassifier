import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing

from custom_transforms import CustomTransformation, AddGaussianNoise
from generate_targets import TargetGenerator

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LiveSegmentDataset(Dataset):
    def __init__(self, length, input_size, target_size, expansion_factor=1, fill_prob=0.5,
            bkg_path=None, target_transforms=None, transforms=None):
        """ Dataset that makes generator object and calls it in __getitem__ """
        self.length = length
        self.input_size = input_size
        self.target_size = target_size
        self.fill_prob = fill_prob
        self.bkg_path = bkg_path
        self.target_transforms = target_transforms
        self.gen = TargetGenerator(input_size=input_size, target_size=target_size,
            expansion_factor=expansion_factor, target_transforms=target_transforms, bkg_path=bkg_path)
        self.transform_mask = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx, transforms=None):
        if transforms == None:
            transforms = self.transforms
        img, mask = self.gen.gen_segment(fill_prob=self.fill_prob)
        return self.transforms(img), self.transform_mask(mask)


def visualize_dataloader(dataloader):
    from torchvision.utils import make_grid
    out = None
    nrows = 6
    rows = []
    for i in range(nrows):
        row = []
        for images, masks in dataloader:
            for i in range(images.shape[0]):
                row.append(images[i].detach().cpu().numpy().transpose(1, 2, 0))
            break
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows) * 255
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/segment_processed.png")


def dataset_stats(dataset, num=1000):
    mean = 0.0
    var = 0.0
    n = min(len(dataset), num)  # Go through the whole dataset if possible
    t0 = time.time()
    t1 = t0
    for i in range(n):
        # img in shape [W, H, C]
        img, y = dataset[i]  # __getitem__
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std
        if (i+1) % 100 == 0:
            t2 = time.time()
            print("{}/{} measured. Total time={:.2f}s. Images per second {:.2f}.".format(i+1, n, t2-t0, 100/(t2-t1)))
            t1 = t2
    print("mean :", mean/n)
    print("var :", var/n)
    print("std :", np.sqrt(var/n))


def time_dataloader(dataset, batch_size=64, max_num_workers=8, num=4096):
    import psutil
    print("Time Dataloader")
    results = []
    ram = []
    n = min(len(dataset), num)
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(n))
    for i in range(max_num_workers+1):
        print("Running with {} workers".format(i))
        ram_before = psutil.virtual_memory()[3]
        train_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=i, drop_last=True, persistent_workers=(True if i >0 else False))
        max_ram = 0
        ts = time.time()
        [_ for _ in train_loader]
        print("Warmup epoch :", time.time()-ts)
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            r = psutil.virtual_memory()[3]
            if r > max_ram:
                max_ram = r
        # ram after
        ram_usage = max_ram - ram_before
        duration = time.time()-t0
        results.append(duration)
        ram.append(ram_usage)

    for i in range(len(results)):
        print("{:.2f} seconds with {} workers. {:.3f} seconds/batches. {} batches. {:.2f} GB ram.".format(results[i], i, results[i]/(batch_idx+1), batch_idx+1, ram[i]*1e-9))


if __name__ == "__main__":

    input_size = 256

    train_size = 256
    batch_size = 8
    shuffle = False
    num_workers = 0
    drop_last = True

    dataset_folder = None #'images/classify1'  # root directory that has images and labels.csv, if None targets are made during the training
    bkg_path = 'backgrounds/validate'  # path to background images, None is a random color background
    target_size = 80  # Smallest target size
    fill_prob = 0.5
    expansion_factor = 1  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
    ])

    train_transforms = T.Compose([
        CustomTransformation(),
        T.Resize((input_size)),  # Make shortest edge this size
        T.ToTensor(),
        AddGaussianNoise(std=0.01)
    ])
    val_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.ToTensor(),
    ])

    # Save a dataset in the images folder
    # save_segment("segment1", 10, input_size, target_size, expansion_factor, bkg_path, target_transforms)


    train_dataset = LiveSegmentDataset(length=train_size,
        input_size=input_size, target_size=target_size,
        expansion_factor=expansion_factor, bkg_path=bkg_path, fill_prob=fill_prob,
        target_transforms=target_transforms, transforms=train_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))

    x, y = train_dataset[0]
    print("Image :", x.shape)
    print("Mask :", y.shape)

    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    im = T.ToPILImage(mode='RGB')(x)
    # im.show()

    # visualize_dataloader(train_loader)  # use batch_size = 8

    # dataset_stats(train_dataset, num=1000)

    time_dataloader(train_dataset, batch_size=16, max_num_workers=8, num=256)
