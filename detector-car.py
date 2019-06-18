
import cv2

import glob

import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2


import random
import scipy.io


__PATH__ = './'

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    import pdb;pdb.set_trace()
    boxes = dn.make_network_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res

import sys, os
sys.path.append(os.path.join(os.getcwd(),'darknet','python/'))

import darknet as dn

__PATH_IMG__ = '/data3/Grab_Challenge/'
img_path= 'car_ims'
annotation_file = 'cars_annos.mat'

mat_data = mat = scipy.io.loadmat(__PATH_IMG__ + annotation_file)['annotations'][0]
random.seed(2019)


# Darknet
net = dn.load_net("./darknet/cfg/yolov3_car_test.cfg", "./models/model_detection/yolov3_car_10000.weights", 0)
meta = dn.load_meta("./darknet/cfg/car.data")

def save_image(is_test, img, label,filename):
     if is_test==False:
         if not os.path.exists('./data/cars/train/{}'.format(label) ):
             os.makedirs('./data/cars/train/{}'.format(label))
         cv2.imwrite('./data/cars/train/{}/{}'.format(label,filename),img)
     else:
         if not os.path.exists('./data/cars/val/{}'.format(label) ):
             os.makedirs('./data/cars/val/{}'.format(label))
         cv2.imwrite('./data/cars/val/{}/{}'.format(label,filename),img)

import tqdm
for i in tqdm.tqdm(range(mat_data.shape[0])):
   filename = mat_data[i][0][0].split('/')[-1]
   
   box = [mat_data[i][1][0],mat_data[i][2][0],mat_data[i][3][0],mat_data[i][4][0]]
   label = mat_data[i][5][0][0]
   arr = cv2.imread(os.path.join(__PATH_IMG__,img_path,filename))
   #im = array_to_image(arr)
   #dn.rgbgr_image(im)
   r = dn.detect(net, meta, os.path.join(__PATH_IMG__,img_path,filename))
   is_test = random.random()>=0.8
   if len(r) <=0:
      save_image(is_test,arr,label, filename)
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
      imgroi = arr[int(y):int(y+h),int(x):int(x+w)]
      save_image(is_test,imgroi,label, filename)
      
   
   
