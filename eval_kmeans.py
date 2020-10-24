'''
Script to generate embeddings from resnet trained using pcl
Command to run:

python eval_kmeans.py --pretrained experiment_pcl_resume/checkpoint.pth.tar /home/mprabhud/dataset/shapenet_renders/npys/

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
import faiss

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
    parser.add_argument('--low-dim', default=16, type=int,
                    help='feature dimension (default: 128)')
    parser.add_argument('--pcl-r', default=1024, type=int,
                        help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='softmax temperature')

    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco-v2/SimCLR data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    parser.add_argument('--num-cluster', default='2500,5000,10000', type=str, 
                        help='number of clusters')

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
    args.num_cluster = args.num_cluster.split(',')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ########################################################################
    # STEP 1: SETuP DATALOADER (MAKE SURE TO CONVERT IT TO PIL IMAGE !!!!!)#
    ########################################################################
    
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = pcl.loader.ShapeNet(
        traindir,
        'split_allpt.txt',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*2, shuffle=False,
        sampler=None, num_workers=args.num_workers, pin_memory=True)
    
    ############################
    # STEP 2: INITIALIZE MODEL #
    ############################

    # create model
    print("=> creating model '{}'".format(args.arch))
    kmeans_model = models.__dict__[args.arch](num_classes=16)
    kmeans_model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), kmeans_model.fc)
    

    # load from pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")            
            state_dict = checkpoint['state_dict']
            # rename pre-trained keys
            
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_k'): 
                    # remove prefix
                    state_dict[k[len("module.encoder_k."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]  
            
            kmeans_model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            
    kmeans_model.cuda()
    
    ###############################
    # STEP 3: GET Kmeans Clusters #
    ##############################
    
    cluster_result = None
    features = compute_embeddings(train_loader, kmeans_model, args) #generate embeddings based on keys encoder (different from eval_embeddings.py)
    # placeholder for clustering result
    cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
    for num_cluster in args.num_cluster:
        cluster_result['im2cluster'].append(torch.zeros(len(train_dataset),dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(int(num_cluster),16).cuda())
        cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
        
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()
    cluster_result = run_kmeans(features,args)  #run kmeans clustering 
    
        
    
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


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        print('performing kmeans clustering on ...',num_cluster)
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()

        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0   
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results
    
if __name__ == '__main__':
    main()