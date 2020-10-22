from PIL import ImageFilter
import random
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset


import torch
import numpy as np
import pickle
import os, sys

from PIL import Image

import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index
    
class ShapeNet(Dataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.
    Arguments:
    """

    def __init__(self, file_root, file_list, train=True, transform=None):

        self.file_root = file_root
        self.train = train
        self.transform = transform
        # Read file list
        self.file_names = []
        with open(os.path.join(file_root,file_list), "r") as fp:
            self.temp_names = fp.read().split("\n")[:-1]
        for name in self.temp_names:
            for view_num in range(24):
                self.file_names.append(name + '-' + str(view_num))
        self.file_nums = len(self.file_names)


    def __getitem__(self, index):
        
        #st()
        name = os.path.join(self.file_root, self.file_names[index])
        file_name, view_num = name.split('-')
        data = pickle.load(open(file_name, "rb"))
        images = torch.tensor(data['rgb_camXs_raw']).permute(0,3,1,2)/255.
        sample = images[int(view_num)]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, index


    def __len__(self):
        return self.file_nums