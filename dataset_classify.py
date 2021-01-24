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
        """ Dataset that makes generator object and calls it in __getitem__ """
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
        """ Dataset from a folder at root.
            This corresponds to the output from save_classify()."""
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
    fig.savefig('classify_processed.png', bbox_inches='tight')
    plt.show()


def visualize_labels(dataloader):
    height = 4
    width = 4
    fig, axarr = plt.subplots(height, width)
    np.vectorize(lambda axarr:axarr.axis('off'))(axarr)  # https://stackoverflow.com/questions/34686414/turn-axes-off-for-all-subplots-of-a-figure?rq=1
    for images, labels in dataloader:
        images = images.detach()[:height*width]
        labels = labels.detach()[:height*width]
        break

    def fl(label):
        label = label.tolist()
        out = ""
        if dataloader.dataset.gen.angle_quantization == 8:
            out += "Orientation: " + str(dataloader.dataset.gen.angle_options[int(label[0])]) + "\n"
        else:
            out += "Orientation: " + str(int(label[0])) + "\n"
        out += "Shape: " + str(dataloader.dataset.gen.shape_options[int(label[1])]) + "\n"
        out += "Letter: " + str(dataloader.dataset.gen.letter_options[int(label[2])]) + "\n"
        out += "Shape Color: " + str(dataloader.dataset.gen.color_options[int(label[3])]) + "\n"
        out += "Letter Color: " + str(dataloader.dataset.gen.color_options[int(label[4])])
        return out

    idx = 0
    for i in range(height):
        for j in range(width):
            img = images[idx]
            axarr[i][j].imshow(np.array(img).transpose((1,2,0)))
            axarr[i][j].text(32, 20, fl(labels[idx]), style='italic',
                bbox={'facecolor': 'grey', 'alpha': 1.0}, fontsize=8)
            idx +=1
    fig.savefig('images/classify_labels.png', bbox_inches='tight')
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
    print("mean :", mean/n)
    print("std :", np.sqrt(var/n))


def time_dataloader(dataset, batch_size=64, max_num_workers=8):
    import psutil
    print("Time Dataloader")
    results = []
    ram = []
    for i in range(max_num_workers+1):
        print("Running with {} workers".format(i))
        ram_before = psutil.virtual_memory()[3]
        train_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False,
            num_workers=i, drop_last=True)
        t0 = time.time()
        max_ram = 0
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
        print("{:.2f} seconds with {} workers. {:.2f} seconds per {} batches. {:.3f} GB ram.".format(results[i], i, results[i]/(batch_idx+1), batch_idx+1, ram[i]*1e-9))


def save_classify(gen, name, resolution, num=1024):

    # create folder

    # create dataset folder

    # create metadata frame

    mean = 0.0
    var = 0.0
    t0 = time.time()
    t1 = t0
    for i in range(num):

        # create name and path

        # img in shape [W, H, C]
        img, y = gen.gen_classify()
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std

        # save image

        # save metadata


        if (i+1) % 100 == 0:
            t2 = time.time()
            print("{}/{} measured. Total time={:.2f}s. Images per second {:.2f}.".format(i+1, n, t2-t0, 100/(t2-t1)))
            t1 = t2
    print("mean :", mean/num)
    print("std :", np.sqrt(var/num))

    # save dataset information

    # columns, mean, std

    pass

if __name__ == "__main__":

    input_size = 32
    batch_size = 256
    train_size = 4096
    val_size = 1024
    shuffle = False
    num_workers = 0
    drop_last = True

    dataset_folder = None  # root directory that has images and labels.csv, if None targets are made during the training
    val_split = 0.2  # percentage of dataset used for validation
    bkg_path = None  # path to background images, None is a random color background
    target_size = 30
    scale = (0.8, 1.0)
    rotation = True
    expansion_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    target_tranforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
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
    print("Image :", x.shape)
    print("Label :", y, y.shape)

    # im = T.ToPILImage(mode='RGB')(x)
    # im.show()

    visualize_dataloader(train_loader)

    # visualize_labels(train_loader)

    # dataset_stats(train_dataset, num=2000)

    # time_dataloader(train_dataset, batch_size, max_num_workers=3)

    # TODO
    # save_classify(gen, name, resolution=img_size, num=dataset_size)
