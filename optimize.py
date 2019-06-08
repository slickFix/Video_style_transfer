#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:42:42 2019

@author: siddharth
"""

import functools
import vgg, datetime as dt
import tensorflow as tf, numpy as np,os
import transform_net
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def optimize(content_targets,style_target,content_weight,style_weight,
             tv_weight,vgg_path,epochs=2,print_iterations=1000,batch_size = 4,
             save_path='checkpoint/save/model.ckpt',slow=False,learning_rate = 1e-3,debug=False):
    
    if slow:
        batch_size=1
        
    # trimming the total training set size
    extra_train_img = len(content_targets)%batch_size
    
    if extra_train_img>0:
        print("Leaving out {} extra (modulus) train examples ".format(extra_train_img))
        content_targets = content_targets[:-extra_train_img]
    
    train_batch_shape = (batch_size,256,256,3)
    
    style_img_gram_features = {}
    
    # appending the batch size for the style image
    style_img_shape = (1,)+style_target.shape
    
    
    # precomputing the style image gram matrix features
    with tf.Graph().as_default(),tf.device('/cpu:0'),tf.Session() as sess:
        
        # defining style image placeholder and preprocessing the image
        style_image_ph = tf.placeholder(tf.float32,shape=style_img_shape,name='style_image_ph')
        style_image_pre = vgg.preprocess(style_image_ph)
        
        # passing the "preproccessed style image" through the VGG19 network
        style_net = vgg.net(vgg_path,style_image_pre)
        
        # creating the numpy array of the style image
        style_img_feed = np.array([style_target])
        
        for layer in STYLE_LAYERS:
            # activations for the style image's different VGG19 layers
            activations = style_net[layer].eval(feed_dict = {style_image_ph:style_img_feed})
            
            activations = np.reshape(activations,(-1,activations.shape[3]))
            gram = np.matmul(activations.T,activations)/activations.size
            
            style_img_gram_features[layer] = gram
            
    # defining graph for computing the Content cost, Style cost and TV cost
    with tf.Graph().as_default(),tf.Session() as sess:
        
        X_content_ph = tf.placeholder(tf.float32,shape =train_batch_shape,name='X_content_ph' )
        X_pre = vgg.preprocess(X_content_ph)
        
        # precomputing the content image activation for content loss
        content_img_activation = {}
        
        content_net = vgg.net(vgg_path,X_pre)
        content_img_activation[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        
        if slow :
            preds = tf.Variable(tf.random_normal(X_content_ph.get_shape())*0.256)
            preds_pre = preds
        
        else:
            # getting the generated image by transforming the content image
            
            preds = transform_net.net(X_content_ph/255.0)
            preds_pre = vgg.preprocess(preds)
         
        # passing the "preproccessed generated image" through the VGG19 network    
        gen_net = vgg.net(vgg_path,preds_pre)
        
        assert _tensor_size(content_img_activation[CONTENT_LAYER]) == _tensor_size(gen_net[CONTENT_LAYER])
                    
        # calculating the content loss
        content_img_size = _tensor_size(content_img_activation[CONTENT_LAYER])*batch_size
        content_loss = content_weight * ( 2 * 
             tf.nn.l2_loss(gen_net[CONTENT_LAYER]-content_img_activation[CONTENT_LAYER])/content_img_size
                )            
        
        