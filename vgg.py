#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:13:59 2019

@author: siddharth
"""

import tensorflow as tf
import numpy as np
import scipy.io

MEAN_PIXEL = np.array([123.8,116.779,103.939])

def net(vgg_path,input_image):
    
    '''
    Gets all the layer's activatinos for the input_image after 
    passing it through the VGG network
    '''
    
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
        )
    
    # loading the vgg matlab model
    # data is of dict type
    data = scipy.io.loadmat(vgg_path)
    
    ''' 
    data['normalization'] =
    [[(array([[[123.68 , 116.779, 103.939],
        [123.68 , 116.779, 103.939],
        [123.68 , 116.779, 103.939],
        ...
        [123.68 , 116.779, 103.939],
        [123.68 , 116.779, 103.939],
        [123.68 , 116.779, 103.939]]]), 
        array([[1.]]), array([[0., 0.]]), array([[224., 224.,   3.]]), array(['bilinear'], dtype='<U8'))]]
    '''
    
    # calculating the mean pixel
    mean = data['normalization'][0][0][0] # mean shape is (224,224,3)
    mean_pixel = np.mean(mean,axis = (0,1)) # mean_pixel shape is (3,)
    
    # getting all the weights 
    weights = data['layers'][0] # weights shape is (43,)
    
    net = {}
    current = input_image
    
    for i,name in enumerate(layers):
        layer_type = name[:4]
        
        if layer_type == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            
            kernels = np.transpose(kernels,(1,0,2,3))
            bias = bias.reshape(-1)
            current = _conv_layer(current,kernels,bias)
            
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        
        elif layer_type == 'pool':
            current = _pool_layer(current)
        
        net[name] = current
    
    assert len(net) == len(layers)
    return net
    

def _conv_layer(input,weights,bias):
    ''' performs convolution operation on the input'''
    conv = tf.nn.conv2d(input,tf.constant(weights),strides = (1,1,1,1),padding='SAME')
    return tf.nn.bias_add(conv,bias)

def _pool_layer(input):
    '''performs pooling operation on the input '''
    
    # ksize: A list or tuple of 4 ints. The size of the window for each dimension of the input tensor.
    return tf.nn.max_pool(input,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')

def preprocess(image):
    return image-MEAN_PIXEL

def unprocess(image):
    return image+MEAN_PIXEL
    