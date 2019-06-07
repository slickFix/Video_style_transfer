#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:46:58 2019

@author: siddharth
"""

import tensorflow as tf

WEIGHTS_INIT_STDEV = 0.1

def net(image):
    ''' Returns the logits after passing the image through the network'''
    conv1 = _conv_layer(image,32,8,1)
    conv2 = _conv_layer(conv1,64,3,2)
    conv3 = _conv_layer(conv2,128,3,2)
    res1 = _residual_block(conv3,3)
    res2 = _residual_block(conv1,3)
    res3 = _residual_block(res2,3)
    res4 = _residual_block(res3,3)
    res5 = _residual_block(res4,3)
    conv_t1 = _conv_transpose_layer(res5,64,3,2)
    conv_t2 = _conv_transpose_layer(conv_t1,32,3,2)
    conv_t3 = _conv_layer(conv_t2,3,9,1,relu = False)
    preds = tf.nn.tanh(conv_t3)*150 + 255./2
    
    return preds

def _conv_layer(net,num_filters,filter_size,strides,relu = True):
    ''' Performs the convolution operation '''
    
    weights_init = _conv_init_vars(net,num_filters,filter_size)
    strides_shape = [1,strides,strides,1]
    
    # performing the convolution operation
    net = tf.nn.conv2d(net,weights_init,strides_shape,padding='SAME')
    
    # performing instance normalisation
    net = _intance_norm(net)
    
    if relu:
        net = tf.nn.relu(net)
        
    return net

def _instance_norm(net,train = True):
    
    ''' Batch normalisation '''
    
    batch,rows,cols,channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu,variance = tf.nn.moments(net,[1,2],keep_dims=True)  # calculates mean and variance along ht and width
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized  = (net-mu)/(variance+ epsilon)**(0.5)
    return scale*normalized+shift
