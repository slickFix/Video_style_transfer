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

from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 4
DEVICE = '/gpu:0'


def ffwd_video(path_in,path_out,checkpoint_dir,device_t = '/gpu:0',batch_size = 4):
    ''' feed forward video '''
    
    # defining video rendering variables
    video_clip = VideoFileClip(path_in,audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out,video_clip.size,video_clip.fps,codec = 'libx264',
                                                    preset = 'medium',bitrate = '2000k',
                                                    audiofile =path_in,threads=None,
                                                    ffmpeg_params=None)
    
    # defining tensorflow variables 
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement = True)
    soft_config.gpu_options.allow_growth = True
    
    # starting the tensorflow session
    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        
        batch_shape = (batch_size,video_clip.size[1], video_clip.size[0], 3)
        
        # defining placeholder
        vid_ph = tf.placeholder(tf.float32,shape=batch_shape,name='vid_ph')
        
        # forward propogation (building the graph)
        preds = transform_net.net(vid_ph)
        
        # defining saver
        saver = tf.train.Saver()
        
        # restoring the saved model
        
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)
        
        x = np.zeros(batch_shape,dtype = np.float32)
        
        # function to generate styled video (batch images) and writing
        def style_and_write(count):
            
            # for batch size not complete case
            for i in range(count,batch_size):
                x[i] = x[count-1] # using last frame(received from .iter_frames) to fill remaing x (batch size not complete case)
            
            # running the graph to style video
            _preds = sess.run(preds,feed_dict = {vid_ph:x})
            
            for i in range(0,count):
                video_writer.write_frame(np.clip(_preds,0,255).astype(np.uint8))
                
        frame_count = 0 # the frame count written to x
        
        for frame  in video_clip.iter_frames():
            
            x[frame_count] = frame
            frame_count +=1
            
            if frame_count == batch_size:
                
                style_and_write(frame_count)
                frame_count = 0
        
        # for last batch where no of images is less than the batch_size                
        if frame_count !=0:
            style_and_write(frame_count)
            
        video_writer.close()

def ffwd_img(data_in,paths_out,checkpoint_dir,device_t='/gpu:0',batch_size = 4):
    ''' feed forward image function '''
    assert len(paths_out)>0
    
    # checking if the input path is string type
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert  len(data_in) == len(paths_out)
        
        # if called via 'ffwd_different_dimensions" shape remains same for that call
        img_shape = get_img(data_in[0]).shape
    else:
        print("Input path is not string, aborting ")
        return
    
    
    batch_size  = min(len(paths_out),batch_size)
    
    # defining tensorflow parameters
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement = True)
    soft_config.gpu_options.allow_growth = True
    
    # Starting the tf session
    with g.as_default(),g.device(device_t),tf.Session(config = soft_config) as sess:
        
        # appending batch size 
        batch_shape = (batch_size,) + img_shape
        
        # defining placeholders
        img_placeholder = tf.placeholder(tf.float32,shape = batch_shape,name = 'img_placeholder')
        
        # forward propogation i.e. defining the network
        preds = transform_net.net(img_placeholder)
        
        saver = tf.train.Saver()
        
        # restoring the trained network
        if os.path.isdir(checkpoint_dir):
            
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)
        
        # defining the no of iterations
        num_iters = int(len(paths_out)/batch_size)
        
        for i in range(num_iters):
            
            # defining the start and end position of the batch
            start_pos = i*batch_size
            end_pos = start_pos + batch_size
            
            curr_batch_out = paths_out[start_pos:end_pos]
            
            # creating the feedable input
            if is_paths:
                curr_batch_in = data_in[start_pos:end_pos]
                
                x = np.zeros(batch_shape,dtype = np.float32)
                
                for j,path_in in enumerate(curr_batch_in):
                    
                    # reading the image
                    img = get_img(path_in)
                    
                    assert img.shape == img_shape,'Images have different dimensions. ' +  \
                                                    'Resize images or use --allow-different-dimensions.'
                    
                    x[j] = img
            
            else:
                x = data_in[start_pos:end_pos]
                
            # running the model for prediction            
            _preds = sess.run(preds,feed_dict={img_placeholder:x})
            
            # saving the predicted(styled) image
            for j,path_out in enumerate(curr_batch_out):
                save_img(path_out,_preds[j])
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = path_out[num_iters*batch_size:]
    
    if len(remaining_in) > 0:
        ffwd_img(remaining_in,remaining_out,checkpoint_dir,device_t=device_t,batch_size=1)
                    
        

def ffwd_different_dimensions(in_path,out_path,checkpoint_dir,device_t=DEVICE,batch_size = BATCH_SIZE):
    
    ''' feed forward with diffrent dimension images '''
    
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        
        shape = '%dx%dx%d'% get_img(in_image).shape
        
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
        
    for shape in in_path_of_shape:
        # process all the image of same dimensions
        print('Processing images of shape %s '%shape)
        
        ffwd_img(in_path_of_shape[shape],out_path_of_shape[shape],
                 checkpoint_dir,device_t,batch_size)

def ffwd_single_img(in_path,out_path,checkpoint_dir,device='/cpu:0'):
    
    ''' feed forward with single image'''
    paths_in,paths_out = [in_path],[out_path]
    
    ffwd_img(paths_in,paths_out,checkpoint_dir,batch_size = 1,device_t=device)

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


def validate_options(options):
    
    assert os.path.exists(options.checkpoint_dir),'checkpoint not found!'
    assert os.path.exists(options.in_path),'Input path not found!'
    
    if os.path.isdir(options.out_path):
        assert os.path.exists(options.out_path),'output dir not found'
    
    assert options.batch_size >0
    

def main():
    # creating input argument parser
    parser = build_parser()
    
    options = parser.parse_args()
    
    validate_options(options)
    
    # if input path is file
    if not os.path.isdir(options.in_path):
        # if output path exists and is directory
        if os.path.exists(options.out_path) and os.path.isdir(options.out_path):
            out_path = \
                        os.path.join(options.out_path,os.path.basename(options.in_path))
        
        else:
            out_path = options.out_path
            
        ffwd_single_img(options.in_path,out_path,options.checkpoint_dir,device = options.device)
    
    # if input path is directory
    else:
        files = list_files(options.in_path)
        full_in = [os.path.join(options.in_path,x) for x in files]
        full_out = [os.path.join(options.out_path,x) for x in files]
        
        if options.allow_different_dimensions:
            ffwd_different_dimensions(full_in,full_out,options.checkpoint_dir,
                                      device_t = options.device,batch_size = options.batch_size)
            
        else:
            ffwd_img(full_in,full_out,options.checkpoint_dir,device_t=options.device,
                 batch_size = options.batch_size)

if __name__ == '__main__':
    main()    