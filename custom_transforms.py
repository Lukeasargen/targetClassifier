import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
import torchvision.transforms.functional as TF  # Image processing

from helper import pil_loader


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=max(np.random.normal(loc=0.0, scale=0.1), 0.0)))
        return img


class RandomGamma(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_gamma(img, gamma=np.random.normal(loc=1.0, scale=0.03))
        return img


class RandomBrightness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_brightness(img, brightness_factor=np.random.normal(loc=1.0, scale=0.01))
        return img

class RandomSharpness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(np.random.normal(loc=0.0, scale=0.03), 0.0)))
        return img


class CustomTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = RandomGaussianBlur(p=1.0)
        self.gamma = RandomGamma(p=1.0)
        self.brightness = RandomBrightness(p=1.0)
        self.sharpen = RandomSharpness(p=1.0)

    def forward(self, img):
        # img = self.gaussian(img)
        img = self.gamma(img)
        # img = self.brightness(img)
        # img = self.sharpen(img)
        return img


if __name__ == "__main__":

    sample_path = "dev/visualize_classify.png"
    sample_img = pil_loader(sample_path)    

    transform = CustomTransformation()

    out = transform(sample_img)

    out.show()
    