#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:03:34 2019

@author: siddharth
"""

import scipy.misc, numpy as np,os

def save_img(out_path,img):
    img = np.clip(img,0,255).astype(np.uint8)
    scipy.misc.imsave(out_path,img)
    

def scale_img(style_path,style_scale):
    o0,o1,o2 = scipy.misc.imread(style_path,mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0*scale),int(o1*scale),o2)
    style_target = get_img(style_path,img_size=new_shape)
    return style_target


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