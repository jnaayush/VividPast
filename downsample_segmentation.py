#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:36:38 2018

@author: ryangreen
"""
import numpy as np

#seg = np.load("segmentation_arrays/results_train_0.npy", mmap_mode='r')

# downsample and save segmentation arrays



for i in range(10):
    
    if i == 0:
        seg = np.load("segmentation_arrays/results_train_" + str(0) + ".npy", mmap_mode='r')
        seg = seg[:,9::18, 9::18,:]
        s = seg
    else:
        print("S:", s.shape)
        seg = np.load("segmentation_arrays/results_train_" + str(i*100) + ".npy", mmap_mode='r')
        seg = seg[:,9::18, 9::18,:]
        s = np.concatenate([s,seg], axis=0)
        print("new S", s.shape)

np.save('seg_data_downsampled_1000', s)
