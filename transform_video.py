#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:20 2019

@author: siddharth
"""

from argparse import ArgumentParser
import os,random,evaluate
from utils import list_files


TMP_DIR = '.fast_neural_style_frames_%s/' % random.randint(0,99999)
DEVICE = '/gpu:0'
BATCH_SIZE = 4

def build_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--checkpoint',type = str,
                        dest = 'checkpoint',help = 'checkpoint directory or .ckpt file',
                        metavar = 'CHECKPOINT', required = True)
    
    parser.add_argument('--in-path',type = str,
                        dest='in_path',help = 'input video path',
                        metavar = 'IN_PATH',required = True)
    
    parser.add_argument('--out-path',type = str,
                        dest='out',help = 'path to save the generated video to',
                        metavar = 'OUT',required = True)
    
    parser.add_argument('--tmp-dir',type = str,
                        dest='tmp_dir',help = 'tmp dir for processing ',
                        metavar = 'TMP_DIR',default = TMP_DIR)
    
    parser.add_argument('--device',type=str,
                        metavar = 'device',help = 'device for eval. gpu = /gpu:0',
                        metavar = 'DEVICE',default = DEVICE)
    
    parser.add_argument('--batch-size',type = int,
                        dest = 'batch_size',help = 'batch size for eval. default is 4',
                        metavar = 'BATCH_SIZE',default= BATCH_SIZE)
    
    
    return parser


    
def main():
    parser = build_parser()

if __name__ == '__main__':
    main()