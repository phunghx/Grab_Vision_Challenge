import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np

import torch.nn.functional as Fn

import pandas as pd
from torchvision import datasets
from functions import *
from imagepreprocess import *
from model_init import *
from src.representation import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import load_data


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--representation', default='MPNCOV', type=str,
                    help='define the representation method')
parser.add_argument('--num-classes', default=196, type=int,
                    help='define the number of classes')
parser.add_argument('--freezed-layer', default=0, type=int,
                    help='define the end of freezed layer')
parser.add_argument('--classifier-factor', default=5, type=int,
                    help='define the multiply factor of classifier')




def main():
    global args
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    
    # create model
    if args.representation == 'GAvP':
        representation = {'function':GAvP,
                          'input_dim':2048}
    elif args.representation == 'MPNCOV':
        representation = {'function':MPNCOV,
                          'iterNum':5,
                          'is_sqrt':True,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':None }
    elif args.representation == 'BCNN':
        representation = {'function':BCNN,
                          'is_vec':True,
                          'input_dim':2048}
    elif args.representation == 'CBP':
        representation = {'function':CBP,
                          'thresh':1e-8,
                          'projDim':8192,
                          'input_dim': 512}
    else:
        warnings.warn('=> You did not choose a global image representation method!')
        representation = None # which for original vgg or alexnet

    model = get_model(args.arch,
                      representation,
                      args.num_classes,
                      args.freezed_layer,
                      )
    
    
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            
            
            model.load_state_dict(checkpoint['state_dict'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    
    
    
    
    print("=> start evaluation")
    
    
    validate(model)

import scipy.io
class_names = scipy.io.loadmat('../cars_annos.mat')['class_names'][0]

def validate( model):
    batch_time = AverageMeter()
    end = time.time()
    # switch to evaluate mode
    model.eval()
    indexs = []
    result_scores = {}
    for i in range(class_names.shape[0]):
         result_scores[class_names[i][0]] = []
    result_preds = []
    with torch.no_grad():
        end = time.time()
        for fnames, input in load_data.test_data_generator(args.batch_size,args.data):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            

            # compute output
            ## modified by jiangtao xie
            if len(input.size()) > 4:# 5-D tensor
                bs, crops, ch, h, w = input.size()
                output = model(input.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = model(input)
            
            output = Fn.softmax(output,dim=1).cpu().numpy()
            
            preds = output.argmax(axis=1)
            
            indexs.extend(fnames)
            
            for i in range(preds.shape[0]):
                 result_preds.append(class_names[int(load_data.class_names[preds[i]])-1][0])
                 for j in range(class_names.shape[0]):
                     result_scores[class_names[int(load_data.class_names[j])-1][0]].append(output[i][j])
            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()
    class_names_list = [x[0] for x in class_names]
    df = pd.DataFrame(columns=['filename','prediction'] + class_names_list )
    df['filename'] = indexs
    df['prediction'] = result_preds
    for item in class_names_list:
        df[item] = result_scores[item]
    df.to_csv(os.path.join('../submission', 'final_prediction.csv'), index=False)  

    print('Time processing {}'.format(time.time() - end))        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == '__main__':
    main()
