#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:46:58 2019

@author: siddharth
"""

import tensorflow as tf

WEIGHTS_INIT_STDEV = 0.1

def net(image):
    ''' Returns the predictions after passing the image through the network'''
    conv1 = _conv_layer(image,32,8,1)
    conv2 = _conv_layer(conv1,64,3,2)
    conv3 = _conv_layer(conv2,128,3,2)
    res1 = _residual_block(conv3,3)
    res2 = _residual_block(res1,3)
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
    net = _instance_norm(net)
    
    if relu:
        net = tf.nn.relu(net)
        
    return net

def _conv_transpose_layer(net,num_filters,filter_size,strides):
    
    ''' Retreiving spatial dim by transpose convolutions '''
    
    # weights transposed (as given in tf documentation)
    weights_init = _conv_init_vars(net,num_filters,filter_size,transpose=True)
    
    batch_size, rows,cols,in_channels = [i.value for i in net.get_shape()]
    output_rows, output_cols = int(rows*strides),int(cols * strides)
    
    output_shape = [batch_size,output_rows,output_cols,num_filters]
    tf_out_shape = tf.stack(output_shape)
    strides_shape = [1,strides,strides,1]
    
    net = tf.nn.conv2d_transpose(net,weights_init,tf_out_shape,strides_shape,padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)
    
    

def _residual_block(net,filter_size=3):
    
    ''' Residual block convolutions '''
    tmp = _conv_layer(net,128,filter_size,1)
    return net + _conv_layer(tmp,128,filter_size,relu = False)

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

def _conv_init_vars(net,out_channels,filter_size,transpose=False):
    
    ''' Performs convolution weights initialisation '''
    
    _,rows,cols,in_channels = [i.value for i in net.get_shape()]
    
    if not transpose:
        weights_shape = [filter_size,filter_size,in_channels,out_channels]
        
    else:
        weights_shape = [filter_size,filter_size,out_channels,in_channels]
        
    weights_init = tf.Variable(tf.truncated_normal(weights_shape,stddev=WEIGHTS_INIT_STDEV,seed=1),dtype = tf.float32)
    
    return weights_init
