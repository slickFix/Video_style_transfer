#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:54:20 2019

@author: siddharth
"""


import os
import numpy as np,scipy.misc
from argparse import ArgumentParser


# =============================================================================
# # defining default options
# =============================================================================
CONTENT_WT = 7.5e0
STYLE_WT = 1e2
TV_WT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH  = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2019'
BATCH_SIZE = 4
DEVICE = '/gpu:0'
FRAC_GPU = 1


def build_parser():
    '''
    Building the parser with different options and hyper parameters
    '''
    parser = ArgumentParser()
    
    parser.add_argument('--checkpoint-dir',type = str,
                        dest='checkpoint_dir',help = 'dir to save checkpoint in',
                        metavar = 'CHECKPOINT_DIR',required = True)
    
    parser.add_argument('--style-path',type =str,
                        dest='style_path',help ='style image path',
                        metavar = 'STYLE',required = True)
    
    parser.add_argument('--train-path',type = str,
                        dest='train_path',help = 'path to training images folder',
                        metavar = 'TRAIN_PATH',default=TRAIN_PATH)
    
    parser.add_argument('--test-path',type =str,
                        dest = 'test_path', help = 'test image path',
                        metavar = 'TEST_PATH',default = False)
    
    parser.add_argument('--test-dir',type =str,
                        dest='test_dir',help = 'test image save dir',
                        metavar = 'TEST_DIR',default = False)
    
    parser.add_argument('--slow',dest = 'slow',action = 'store_true',
                        help = 'gatys\' approach (for debugging, not supported)',
                        default = False)
    
    parser.add_argument('--epochs',type = int,
                        dest = 'epochs',help = 'num epochs',
                        metavar = 'EPOCHS',default=NUM_EPOCHS)
    
    parser.add_argument('--batch_size',type = int,
                        dest = 'batch_size',help = 'batch_size',
                        metavar = 'BATCH_SIZE',default = BATCH_SIZE)
    
    parser.add_argument('--checkpoint-iterations',type = int,
                        dest = 'checkpoint_iterations', help = 'checkpoint frequency',
                        metavar = 'CHECKPOINT_ITERATIONS',default= CHECKPOINT_ITERATIONS)
    
    parser.add_argument('--vgg-path',type = str,
                        dest='vgg_path',help='path to VGG19 network (default %(default)s)',
                        metavar = 'VGG_PATH',default = VGG_PATH)
    
    parser.add_argument('--content-weight',type = float,
                        dest = 'content_weight',help = 'content weight (default %(default)s)',
                        metavar ='CONTENT_WEIGHT',default = CONTENT_WT)
    
    parser.add_argument('--style-weight',type =float,
                        dest = 'style_weight',help ='style weight (default %(default)s)',
                        metavar = 'STYLE_WEIGHT',default = STYLE_WT)
    
    parser.add_argument('--tv-weight',type = float,
                        dest = 'tv_weigth',help = 'total variation regularization weight (default %(default)s)',
                        metavar = 'TV_WEIGHT',default = TV_WT)
    
    parser.add_argument('--learning-rate',type = float,
                        dest='learning_rate',help = 'learning rate (default %(default)s)',
                        metavar = 'LEARNING_RATE',default = LEARNING_RATE)
    
    return parser
    
    
def validate_options(options):
    ''' validating the paths and sanity checking the hyperparameters values'''
    
    assert os.path.exists(options.checkpoint_dir),"checkpoint dir not found!"
    assert os.path.exists(options.style_path),'style path not found !!'
    assert os.path.exists(options.train_path),'train path not found !!'
    
    if options.test_path or options.test_dir:
        assert os.path.exists(options.test_path),'test image not found !!'
        assert os.path.exists(options.test_dir),'test directory not found !!'
    
    assert os.path.exists(options.vgg_path),'vgg network data not found !!'
    
    assert options.epochs>0
    assert options.batch_size >0
    assert options.checkpoint_iterations >0
    assert options.content_weight >=0
    assert options.style_weight >=0
    assert options.tv_weight >=0
    assert options.learning_rate >=0
    

def main():
    # getting the defined parser
    parser = build_parser()
    options = parser.parse_args()
    
    validate_options(options)
    
    

if __name__ == '__main__':
    main()