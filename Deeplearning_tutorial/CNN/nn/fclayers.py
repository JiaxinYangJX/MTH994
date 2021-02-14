# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2021-02-08 22:49:26
LastModifiedBy: Rui Wang
LastEditTime: 2021-02-09 19:32:02
Email: wangru25@msu.edu
FilePath: /CNN/nn/fclayers.py
Description: 
'''
import math
import numpy as np
import pandas as pd
'''
: Fullyconnect Layer:
: z : input, with shape (N, C, H, W), in this project, (N,1,28,28)
      N: #of Sampels
      C: input channels, actually the number of colors, usually C = 0,or 1, or 2
      H: Height of input figure
      W: Width of input figure
: next_dz : The gradient of the output Convolutional layer, with shape (N, D, H', W')
      N: #of Sampels
      D: #of filters
      H': Height
      W': Width
: W : Weights, with shape (C, D, k1, k2), in this project, (1,filters, 3,3)
      C: #of samples
      D: output channels, actually #of filters
      k1: height of Weights
      k2: width of Weights
: b : bias, with shape (D,)
      D: output channels, actually #of filters
'''

def fc_forward(z, W, b):
    """
    : Fullyconnect Forward process
    """
    return np.dot(z, W) + b

def fc_backward(next_dz, W, z):
    """
    : Fullyconnect Backpropogation process
    """
    N = z.shape[0]
    dz = np.dot(next_dz, W.T) # gradient of current layer
    dW = np.dot(z.T, next_dz)
    db = np.sum(next_dz, axis=0, keepdims=True)

    return dW/N, db/N, dz
