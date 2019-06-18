import os.path
import glob

import pickle
import os
from os import listdir, getcwd
from os.path import join

from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from PIL import Image

import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)





random.seed(2019)

__PATH_DETECTION__ = '../../data/data_detection/'
__PATH_IMG__ = '/data3/Grab_Challenge/'
train_path = 'train/'
val_path = 'val/'
img_path= 'car_ims'
annotation_file = 'cars_annos.mat'
roi_train = 'train_roi/'
roi_val = 'val_roi/'

make_sure_path_exists(os.path.join(__PATH_DETECTION__,roi_train))
make_sure_path_exists(os.path.join(__PATH_DETECTION__,roi_val))
from shutil import copyfile

def convert_annotation(filename, box,  folder_roi, folder_img):

    image_path = os.path.join(__PATH_IMG__,img_path,filename)
    copyfile(image_path, os.path.join(__PATH_DETECTION__, folder_img,filename))
    out_file = open(os.path.join(__PATH_DETECTION__ ,folder_roi,filename.replace('jpg','txt')) , 'w')

    img = Image.open(image_path)
    
    
    w,h = img.size
    #img = np.array(img,dtype=np.uint8)
    flag = False
    cls_id = 0
    b = (float(box[0]),float(box[2]), float(box[1]), float(box[3]))
        
    bb = convert((w,h), b)
    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    flag=True
    return flag

import os
import os.path

train_files = []
valid_files = []

mat_data = mat = scipy.io.loadmat(__PATH_IMG__ + annotation_file)['annotations'][0]

list_file_train = open(__PATH_DETECTION__ + 'cars_train.txt', 'w')
list_file_test = open(__PATH_DETECTION__ + 'cars_test.txt', 'w')

import tqdm
for i in tqdm.tqdm(range(mat_data.shape[0])):
   filename = mat_data[i][0][0].split('/')[-1]
   
   box = [mat_data[i][1][0],mat_data[i][2][0],mat_data[i][3][0],mat_data[i][4][0]]
   is_test = random.random()>=0.8
   if is_test == False:
       convert_annotation(filename,box,roi_train,train_path)
       list_file_train.write(os.path.join(__PATH_DETECTION__, train_path,filename + '\n'))
   else:
       convert_annotation(filename,box,roi_val,val_path)
       list_file_test.write(os.path.join(__PATH_DETECTION__, val_path,filename + '\n'))
    

list_file_train.close()
list_file_test.close()

