#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:12:19 2018

@author: ryangreen
"""

import numpy as np

### compute normalization constants to use when rescaling data
def getScalingFactors(total_train_samples, num_test_samples):
    L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    L_channel = np.expand_dims(L_channel, axis=3)
    
    L_mu = np.mean(L_channel, axis=(0,1,2)) 
    L_half_std = np.std(L_channel, axis=(0,1,2)) / 2
    
    AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    
    AB_mu = np.mean(AB_channel, axis=(0,1,2))
    AB_half_std = np.std(AB_channel, axis=(0,1,2)) / 2
    
    seg_data_raw = np.load("segmentation_arrays/segmentation_data_5000.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    seg_factor =  np.max(seg_data_raw, axis=(0,1,2))
    
    return {"L_mu": L_mu,
            "L_factor": L_half_std,
            "AB_mu": AB_mu,
            "AB_factor": AB_half_std,
            "seg_factor": seg_factor}
    
np.save('scaling-factors-all-data.npy', getScalingFactors(1000, 50))
    
#test = getScalingFactors(1000, 50)