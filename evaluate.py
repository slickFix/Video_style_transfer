#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:08:12 2019

@author: siddharth
"""

import transform_net,numpy as np,vgg,os
import scipy.misc
import tensorflow as tf
from utils import save_img,get_img,list_files
from argparse import ArgumentParser
from collections import defaultdict
import datetime as dt

BATCH_SIZE = 4
DEVICE = '/gpu:0'


def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--checkpoint',type =str,
                        dest = 'checkpoint_dir',help='dir or .ckpt file to load checkpoint from',
                        metavar = 'CHECKPOINT',required = True)
    
    parser.add_argument('--in-path',type = str,
                        dest='in_path',help = 'dir or file to transform',
                        metavar = 'IN_PATH',required = True)
    
    parser.add_argument('--out-path',type =str,
                        dest = 'out_path',help = 'destination (dir or file) of transformed file or files',
                        metavar = 'OUT_PATH',required = True)
    
    parser.add_argument('--device',type =str,
                        dest = 'device',help = 'device to perform compute on ',
                        metavar='DEVICE',default=DEVICE)
    
    parser.add_argument('--batch-size',type=int,
                        dest='batch_size',help = 'batch size for feedforwarding ',
                        metavar = 'BATCH_SIZE',default=BATCH_SIZE)
    
    parser.add_argument('--allow-different-dimensions',action = 'store_true',
                        dest = 'allow_different_dimensions',help = 'allows different image dimensions')
    
    return parser

def main():
    # creating input argument parser
    parser = build_parser()

if __name__ == '__main__':
    main()    