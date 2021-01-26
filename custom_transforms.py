import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader  # Building custom datasets
import torchvision.transforms as T  # Image processing
import torchvision.transforms.functional as TF  # Image processing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(numpy.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=max(np.random.normal(loc=0.0, scale=0.1), 0.0)))
        return img


class RandomGamma(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(numpy.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_gamma(img, gamma=np.random.normal(loc=1.0, scale=0.03))
        return img


class RandomBrightness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(numpy.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_brightness(img, brightness_factor=np.random.normal(loc=1.0, scale=0.01))
        return img

class RandomSharpness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(numpy.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(np.random.normal(loc=0.0, scale=0.03), 0.0)))
        return img


class CustomTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        c = np.random.uniform(size=4) # slightly faster to random sample 4 at once
        if c[0] < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=max(np.random.normal(loc=0.0, scale=0.1), 0.0)))
        if c[1] < 0.5:
            img = TF.adjust_gamma(img, gamma=np.random.normal(loc=1.0, scale=0.03))
        if c[2] < 0.5:
            img = TF.adjust_brightness(img, brightness_factor=np.random.normal(loc=1.0, scale=0.01))
        if c[3] < 0.1:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(np.random.normal(loc=0.0, scale=0.03), 0.0)))
        return img
