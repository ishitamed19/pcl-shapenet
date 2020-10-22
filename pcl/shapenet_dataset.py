
from torchvision import transforms
from torch.utils.data import Dataset
import pcl

import torch
import numpy as np
import pickle
import os, sys

from PIL import Image

import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize


# import ipdb
# st = ipdb.set_trace

class ShapeNet(data.Dataset):
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
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


    def __len__(self):
        return self.file_nums


def create_imagenet_dataset(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # center-crop augmentation
    eval_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = pcl.loader.ImageFolderInstance(
        traindir,
        pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    eval_dataset = pcl.loader.ImageFolderInstance(
        traindir,
        eval_augmentation)
    return train_dataset, eval_dataset


def create_shapenet_dataset(args):
    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                     std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # center-crop augmentation
    eval_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ShapeNet(
        traindir,
        'split_allpt.txt',
        transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    eval_dataset = ShapeNet(
        traindir,
        'split_allpt.txt',
        transform=eval_augmentation)
    return train_dataset, eval_dataset

if __name__ == "__main__":

    file_root = "/home/mprabhud/dataset/shapenet_renders/npys/"
    dataloader = ShapeNet(file_root, 'split_allpt.txt')

    print("Load %d files!\n" % len(dataloader))
    
    image = dataloader[0]

    print("Info for the first data:")
    print("Image Shape: ", image.shape)