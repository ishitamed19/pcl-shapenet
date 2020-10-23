'''
Script to generate embeddings from resnet trained using pcl
Command to run:

python eval_embeddings.py --pretrained experiment_pcl_crop128_finnetunedorange/checkpoint.pth.tar /home/mprabhud/dataset/shapenet_renders/npys/

'''

from __future__ import print_function

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import random
import numpy as np
from tqdm import tqdm

from torchvision import transforms, datasets
import torchvision.models as models
import pcl.loader

import ipdb
st = ipdb.set_trace


def parse_option():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--cost', type=str, default='0.5')
    parser.add_argument('--seed', default=0, type=int)
    
    # model definition
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')    
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')
    # dataset
    parser.add_argument('--low-shot', default=False, action='store_true', help='whether to perform low-shot training.')    
    
    opt = parser.parse_args()

    opt.num_class = 20
    
    # if low shot experiment, do 5 random runs
    if opt.low_shot:
        opt.n_run = 5
    else:
        opt.n_run = 1
    return opt


def main():
    args = parse_option()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ########################################################################
    # STEP 1: SETuP DATALOADER (MAKE SURE TO CONVERT IT TO PIL IMAGE !!!!!)#
    ########################################################################
    
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                     std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    
    train_dataset = pcl.loader.ShapeNet(
        traindir,
        'split_allpt.txt',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=None, num_workers=args.num_workers, pin_memory=True)
    
    ############################
    # STEP 2: INITIALIZE MODEL #
    ############################

    # create model
    print("=> creating model '{}'".format(args.arch))
    embedding_model = models.__dict__[args.arch](num_classes=16)
    embedding_model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), embedding_model.fc)
    

    # load from pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")            
            state_dict = checkpoint['state_dict']
            # rename pre-trained keys
            
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q'): 
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]  
            
            embedding_model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            
    embedding_model.cuda()
    
    ##########################
    # STEP 3: GET EMBEDDINGS #
    ##########################
    
    embedddings_generated = compute_embeddings(train_loader, embedding_model, args)
    print(embedddings_generated.shape)
    
def compute_embeddings(eval_loader, model, args):
    print('Computing embeddings...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),16).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images) 
            features[index] = feat    
    return features.cpu()
    
if __name__ == '__main__':
    main()