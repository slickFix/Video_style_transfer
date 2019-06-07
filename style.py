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
    
    

def main():
    parser = build_parser()

if __name__ == '__main__':
    main()