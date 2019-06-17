
from torchvision.transforms import functional as F
import glob

#ResNet
import numpy as np

import multiprocessing
import os
import time
import random


from random import getrandbits
import os


import glob
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
import scipy.io
import numpy as np

from PIL import Image
import numpy as np
import torchvision.datasets.folder as torchdataset

class_names = np.loadtxt('classes.txt',dtype=np.str)
num_classes = len(class_names)


n_class = len(class_names)
INPUT_IMAGE_SIZE = 224
INPUT_IMAGE_SIZE_D = 256

max_q_size = 500
maxproc = 12
samples_per_epoch = 320*10
batch_size = 10 * 4
WIDTH = 256
HEIGHT = 256
train_size = 224
channel = 3

    
def random_onoff():                # randomly turns on or off
    return bool(getrandbits(1))


    
def loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

import torch
    
import torchvision.transforms as transforms   
import numbers


def center_crop_with_flip(img, size, vertical_flip=False):
    crop_h, crop_w = size
    first_crop = F.center_crop(img, (crop_h, crop_w))
    if vertical_flip:
        img = F.vflip(img)
    else:
         img = F.hflip(img)
    second_crop = F.center_crop(img, (crop_h, crop_w))
    return (first_crop, second_crop)

class CenterCropWithFlip(object):
    """Center crops with its mirror version.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return center_crop_with_flip(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)   

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
evaluate_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ]) 

#darknet
import sys, os
sys.path.append(os.path.join(os.getcwd(),'../darknet','python/'))
import darknet as dn


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    
    
    def is_valid_file(x):
            return torchdataset.has_file_allowed_extension(x, torchdataset.IMG_EXTENSIONS)
    
    
    if not os.path.isdir(dir):
        return []
    for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path
                    images.append(item)

    return images
    
def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im
import sys, os
sys.path.append(os.path.join(os.getcwd(),'../darknet','python/'))    
net = dn.load_net("../darknet/cfg/yolov3_car_test.cfg", "../models/model_detection/yolov3_car_10000.weights", 0)
meta = dn.load_meta("../darknet/cfg/car.data")

def post_processing(r, img):
    if len(r) <=0:
      return (0,0,img.width-1, img.height-1)
    else:
      index = 0
      c,p,(x,y,w,h) = r[index]
      max_S = w*h
      for j in range(1,len(r)):
          c,p,(x,y,w,h) = r[j]
          if w*h > max_S:
             index = j
             max_S = w*h
      c,p,(x,y,w,h) = r[index]
      x = x-w/2
      y = y-h/2
      x = x if x>=0 else 0
      y = y if y>=0 else 0
      x2 = x+w if (x+w)<img.width else img.width-1
      y2 = y+h if (y+h)<img.height else img.height-1
      return(x,y,x2,y2)
        
def test_data_generator(batch, folders):
    fpaths = make_dataset(folders)
    i = 0
    for path in fpaths:
        
        if i == 0:
            imgs = []
            fnames = []
        filename = path
        
        
        try:
            img_ = loader(filename)
            
            r = dn.detect(net, meta, filename)
        
            box = post_processing(r, img_)
            img_ = img_.crop(box)
        
            img_ =  evaluate_transforms(img_)       
        except:
           print("Error when processing file "+ filename)
           continue

        i += 1
        imgs.append(img_)
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            imgs = torch.stack(imgs)
            yield fnames, imgs
    if i < batch and i!=0 :
        imgs = torch.stack(imgs)
        yield fnames, imgs
    raise StopIteration()


def load_data():
    for X,y in myGenerator(20):
       break
    return X,y
def load_data_val():
    for X,y in valid_data_generator(20):
       break
    return X,y




