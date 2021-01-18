import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
import torchvision.transforms.functional as TF  # Image processing

from generate_targets import TargetGenerator
from custom_transforms import CustomTransformation
from helper import pil_loader


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LiveClassifyDataset(Dataset):
    def __init__(self, length, input_size, target_size, scale=(1.0, 1.0), rotation=False, bkg_path=None,
            target_transforms=None, transforms=None):
        self.length = length
        self.input_size = input_size
        self.target_size = target_size
        self.scale = scale
        self.rotation = rotation
        self.bkg_path = bkg_path
        self.target_transforms = target_transforms
        self.gen = TargetGenerator(input_size, target_size, scale, rotation, target_transforms, bkg_path)
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, y = self.gen.gen_classify()
        return self.transforms(img), torch.tensor(list(y.values()), dtype=torch.float).squeeze()


class FolderClassifyDataset(Dataset):
    def __init__(self, root, transforms):
        self.transforms = T.Compose([T.ToTensor()])
        if transforms == None:
            self.transforms = transforms
        self.root = root
        self.metadata = pd.read_csv(os.path.join(self.root, "metadata.csv"), usecols=range(1))
        self.length = len(self.metadata)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx, 0]  # 0=path, 1:=label
        print(sample)
        label = sample[1:]
        img_path = os.path.join(self.root, sample[0])
        img = pil_loader(img_path)
        print("label :", label)
        # # target needs to be a tensor float of the class index for each task
        # return self.transforms(img), torch.tensor(list(y.values()), dtype=torch.float).squeeze()
        return sample


def visualize_dataloader(dataloader):
    from torchvision.utils import make_grid
    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid((images.detach()[:64]), nrow=8).permute(1, 2, 0))
        break
    plt.show()


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
    print("mean :", mean/num)
    print("std :", np.sqrt(var/num))


def time_dataloader(datalodaer, max_num_workers=4):
    for i in range(max_num_workers+1):
        train_loader = DataLoader(
            dataset=train_dataset
            ,batch_size=batch_size
            ,shuffle=shuffle
            ,num_workers=i
            ,drop_last=drop_last)
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            pass
        print("{:.6f} seconds with {} workers.".format(time.time()-t0, i))


if __name__ == "__main__":

    input_size = 32
    batch_size = 256
    train_size = 16384
    val_size = 1024
    shuffle = False
    num_workers = 0
    drop_last = True

    dataset_folder = None  # root directory that has images and labels.csv, if None targets are made during the training
    val_split = 0.2  # percentage of dataset used for validation
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 30
    scale = (0.6, 1.0)
    rotation = True
    expansion_factor = 4  # generate higher resolution targets and downscale, improves aliasing effects
    target_tranforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=Image.BICUBIC),
    ])

    train_transforms = T.Compose([
        CustomTransformation(),
        T.Resize((input_size)),  # Make shortest edge this size
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.ToTensor(),
    ])

    if dataset_folder:
        # TODO
        # train_dataset = FolderClassifyDataset(root=path, transforms=train_transforms)
        # val_dataset = FolderClassifyDataset(root=path, transforms=val_transforms)
        # load data
        # split
        # make datasets
        pass
    else:
        train_dataset = LiveClassifyDataset(length=train_size, input_size=expansion_factor*input_size,
            target_size=expansion_factor*target_size, scale=scale, rotation=rotation, bkg_path=bkg_path,
            target_transforms=target_tranforms, transforms=train_transforms)
        val_dataset = LiveClassifyDataset(length=val_size, input_size=expansion_factor*input_size,
            target_size=expansion_factor*target_size, scale=scale, rotation=rotation, bkg_path=bkg_path,
            target_transforms=target_tranforms, transforms=val_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
            num_workers=num_workers, drop_last=drop_last)

    x, y = train_dataset[0]
    print(x.shape)
    print(y.shape)

    # im = T.ToPILImage(mode='RGB')(x)
    # im.show()

    visualize_dataloader(train_loader)

    dataset_stats(train_dataset, num=2000)

    # time_dataloader(train_loader, max_num_workers=8)

