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

no_seg_path = 'test_predictions/no_seg/no_seg_small1/predicted_ab_ep68.npy'
seg_path = 'test_predictions/seg/fusion_train1/predicted_ab_ep6301.npy'

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

def get_cost_per_category(true_ab, rescaled_predicted_ab, mask_count):
    costs = calculate_cost(true_ab, rescaled_predicted_ab)
    mask_count = mask_count - 1
    assert len(costs) == len(mask_count)
    
    costs_cat = [[],[],[],[],[],[]]
    
    
    for i in range(len(costs)):
        costs_cat[mask_count[i]].append(costs[i])

        
    return costs_cat

# test arrays
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:100, :, :]
pred_AB_no_seg = np.round((np.load(no_seg_path, mmap_mode='r')+1) * 127.5).astype('uint8')
pred_AB_seg = np.round((np.load(seg_path, mmap_mode='r')+1) * 127.5).astype('uint8')
mask_count = np.load("./image_segmentation/maskcount_combined800.npy")[:100]

costs_no_seg = get_cost_per_category(AB_channel, pred_AB_no_seg , mask_count)
costs_seg = get_cost_per_category(AB_channel, pred_AB_seg , mask_count)

np.save('cost_per_category_no_seg_100.npy', costs_no_seg)
np.save('cost_per_category_seg_100.npy', costs_seg)

print(costs_no_seg)


print(costs_seg)




