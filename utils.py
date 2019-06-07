#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:03:34 2019

@author: siddharth
"""

import scipy.misc, numpy as np,os

def get_img(src,img_size = False):
    img = scipy.misc.imread(src,mode='RGB')
    
    if not (len(img.shape) ==3 and img.shape[2]==3):
        img = np.dstack((img,img,img))
        
    if img_size != False:
        img = scipy.misc.imresize(img,img_size)
        
    return img

def list_files(in_path):
    files = [] 
    for (dir_path,dir_names,file_names) in os.walk(in_path):
        files.extend(file_names)
        break
    
    return files

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]