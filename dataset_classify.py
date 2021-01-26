import json
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

from custom_transforms import CustomTransformation
from generate_targets import TargetGenerator
from helper import pil_loader
from logger import Logger


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LiveClassifyDataset(Dataset):
    def __init__(self, length, input_size, target_size, expansion_factor=1, scale=(1.0, 1.0), rotation=False, bkg_path=None,
            target_transforms=None, transforms=None):
        """ Dataset that makes generator object and calls it in __getitem__ """
        self.length = length
        self.input_size = input_size
        self.target_size = target_size
        self.expansion_factor = expansion_factor
        self.scale = scale
        self.rotation = rotation
        self.bkg_path = bkg_path
        self.target_transforms = target_transforms
        self.gen = TargetGenerator(input_size, target_size, expansion_factor, scale, rotation, target_transforms, bkg_path)
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, y = self.gen.gen_classify()
        return self.transforms(img), torch.tensor(list(y.values()), dtype=torch.float).squeeze()


class FolderClassifyDataset(Dataset):
    def __init__(self, folder, transforms):
        """ Dataset from a folder at root.
            This corresponds to the output from save_classify()."""
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms
        self.folder = folder
        self.metadata = pd.read_csv(os.path.join(self.folder, "labels.csv"))
        self.length = len(self.metadata)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 0]  # 0=path
        label = list(self.metadata.iloc[idx, 1:])  # 1:=label
        img = pil_loader(os.path.join(self.folder, img_name))
        return self.transforms(img), torch.tensor((label), dtype=torch.float).squeeze()


def visualize_dataloader(dataloader):
    from torchvision.utils import make_grid
    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid((images.detach()[:64]), nrow=8).permute(1, 2, 0))
        break
    fig.savefig('images/classify_processed.png', bbox_inches='tight')
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


def save_classify(name, num, input_size, target_size, expansion_factor=1,
        scale=(1.0, 1.0), rotation=False, bkg_path=None, target_transforms=None):
    print("Saving {} images to {} dataset".format(num, name))
    # create generator
    gen = TargetGenerator(input_size, target_size, expansion_factor, scale, rotation, target_transforms, bkg_path)

    # create folder
    folder_path = "images/" + str(name) + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # create metadata frame
    img, y = gen.gen_classify()  # use y for headers
    headers = ["img_name"]
    [headers.append(x) for x in y.keys()]
    logger = Logger(name="labels", headers=headers, date=False, folder=folder_path)

    mean, var = 0.0, 0.0
    t0 = t1 = time.time()
    for i in range(num):
        # create name and path
        img_name = "img_{:07d}.png".format(i)
        row_data = {"img_name": img_name}
        # generate image, calculate stats, and save
        img, y = gen.gen_classify()
        array = np.expand_dims((np.array(img).transpose(2,0,1)), axis=0) / 255
        mean += np.mean(array, axis=(0, 2, 3))
        var += np.var(array, axis=(0, 2, 3))
        img.save(folder_path + img_name)
        # save metadata
        row_data.update(y)
        logger.update(row_data)
        if (i+1) % 100 == 0:
            t2 = time.time()
            print("{}/{} created. Total time={:.2f}s. Images per second {:.2f}.".format(i+1, num, t2-t0, 100/(t2-t1)))
            t1 = t2
    mean = mean/num
    std = np.sqrt(var/num)
    print("mean :", mean)
    print("std :", std)

    # save dataset information
    data = {
        "mean": list(mean),
        "std": list(std)
    }
    # these if statements save time when testing different models
    data.update({"num_classes": gen.num_classes})
    data.update({"input_size": gen.input_size})
    if "orientation" in headers:
        data.update({"orientation": gen.angle_quantization})
    if "shape" in headers:
        data.update({"shape": gen.shape_options})
    if "letter" in headers:
        data.update({"letter": gen.letter_options})
    if "shape_color" in headers:
        data.update({"shape_color": gen.color_options})
    if "letter_color" in headers:
        data.update({"letter_color": gen.color_options})

    json.dump(data, open(folder_path+"set_info.json", 'w'))


if __name__ == "__main__":

    input_size = 32
    val_batch_size=batch_size = 256
    train_size = 4096
    val_size = 1024
    shuffle = False
    num_workers = 0
    drop_last = True

    validate = False
    dataset_folder = 'images/classify0'  # root directory that has images and labels.csv, if None targets are made during the training
    val_split = 0.1  # percentage of dataset used for validation
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 30
    scale = (0.8, 1.0)
    rotation = True
    expansion_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.4, p=1.0, interpolation=Image.BICUBIC),
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

    # Save a dataset in the images folder
    # save_classify("classify0", 50000, input_size, target_size, expansion_factor, scale, rotation, bkg_path, target_transforms)

    if dataset_folder:
        full_set = FolderClassifyDataset(folder=dataset_folder, transforms=train_transforms)
        val_size = int(len(full_set)*val_split)
        if validate:
            if val_size < batch_size:
                val_batch_size = val_size
                print("Validate set is smaller than batch size. val_batch_size is reduced to", val_batch_size)
            train_dataset, val_dataset = torch.utils.data.random_split(full_set, [len(full_set)-val_size, val_size])
        else:
            train_dataset = full_set
        set_info = json.load(open(dataset_folder+"/set_info.json"))
        set_mean = set_info["mean"]
        set_std = set_info["std"]
    else:
        train_dataset = LiveClassifyDataset(length=train_size, input_size=input_size, target_size=target_size,
            expansion_factor=expansion_factor, scale=scale, rotation=rotation, bkg_path=bkg_path,
            target_transforms=target_transforms, transforms=train_transforms)
        if validate:
            val_dataset = LiveClassifyDataset(length=val_size, input_size=input_size, target_size=target_size,
                expansion_factor=expansion_factor, scale=scale, rotation=rotation, bkg_path=bkg_path,
                target_transforms=target_transforms, transforms=val_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last)
    if validate:
        val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size,
                num_workers=num_workers, drop_last=drop_last)

    # x, y = train_dataset[0]
    # print("Image :", x.shape)
    # print("Label :", y, y.shape)

    # im = T.ToPILImage(mode='RGB')(x)
    # im.show()

    # visualize_dataloader(train_loader)

    # visualize_labels(train_loader)  # Does not work for FolderClassifyDataset

    # dataset_stats(train_dataset, num=2000)

    time_dataloader(train_dataset, batch_size, max_num_workers=8, num=8192)
