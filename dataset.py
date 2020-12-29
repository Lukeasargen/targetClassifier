import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
import torchvision.transforms.functional as TF  # Image processing

from generate_samples import TargetGenerator


class CustomTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = img.filter(ImageFilter.GaussianBlur(radius=max(np.random.normal(loc=0.3, scale=0.2), 0.0)))
        img = TF.adjust_gamma(img, gamma=max(np.random.normal(loc=1.0, scale=0.15), 0.5))
        img = TF.adjust_brightness(img, brightness_factor=max(np.random.normal(loc=1.0, scale=0.2), 0.3))
        if np.random.uniform() < 0.1:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(np.random.normal(loc=1.0, scale=0.25), 0.0)))
        return img

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
        img, y = self.gen.draw_target(self.input_size, self.target_size, self.scale, self.rotation)
        return self.transforms(img), torch.tensor(list(y.values()), dtype=torch.float).squeeze()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    input_size = 32
    target_size = 30
    scale = (0.6, 1.0)
    rotation = True

    batch_size = 256
    train_size = 8192
    shuffle = False
    num_workers = 0
    drop_last = True

    set_mean = [0.538, 0.428, 0.381]
    set_std = [0.175, 0.168, 0.191]

    train_transforms = T.Compose([
        CustomTransformation(),
        # T.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC),
        T.Resize((input_size)),  # Make shortest edge this size
        T.ToTensor(),
        # T.Normalize(mean=set_mean, std=set_std)
    ])

    exapnsion_factor = 4  # generate higher resolution targets and downscale, improves aliasing effects
    train_dataset = TargetDataset(transforms=train_transforms, input_size=exapnsion_factor*input_size,
        target_size=exapnsion_factor*target_size, length=train_size, scale=scale, rotation=rotation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size
        ,shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


    # Visualize transforms
    nrows = 4
    ncols = 4
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, label = train_dataset[0]
            row.append(img.numpy().transpose(1, 2, 0)*255)
            # print(i, j, label)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    # im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    # im.show()
    import matplotlib.pyplot as plt
    plt.imshow(grid_img/255.0)
    plt.show()

    exit()

    nimg = 8000
    mean = 0.0
    var = 0.0
    for i in range(nimg):
        # img in shape [W, H, C]
        img, y = train_dataset[0]
        # img = np.random.normal(size=(img_size, img_size, 3)) * 255
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std
    mean = mean/nimg
    std = np.sqrt(var/nimg)
    print("mean :", mean)
    print("std :", std)


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


