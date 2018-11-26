#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:40:44 2018

@author: ryangreen
"""
import numpy as np

# test arrays
#AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:10, :, :]
#pred_AB = np.round((np.load("test_predictions.npy", mmap_mode='r')+1) * 127.5).astype('uint8')
#mask_count = np.load("./image_segmentation/maskcount_combined800.npy")[:10]


# returns array of single cost values (loss) for each image 
def calculate_cost(true_ab, rescaled_predicted_ab):
    assert true_ab.shape == rescaled_predicted_ab.shape
    assert len(true_ab.shape) == 4
    
    diff = np.absolute(true_ab - rescaled_predicted_ab)
    return np.mean(diff, axis=(1,2,3))
    
    
def calculate_cost_per_category(true_ab, rescaled_predicted_ab, mask_count):
    costs = calculate_cost(true_ab, rescaled_predicted_ab)
    mask_count = mask_count - 1
    assert len(costs) == len(mask_count)
    
    sums = np.zeros(6)
    counts = np.zeros(6)
    
    for i in range(len(costs)):
        sums[mask_count[i]] += costs[i]
        counts[mask_count[i]] += 1
    # make sure not to divide by zeros
    for j in range(len(counts)):
        if (counts[j] == 0):
            counts[j] = 1
        
    return sums / counts, counts