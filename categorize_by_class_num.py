#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:48:39 2018

@author: ryangreen
"""
import numpy as np

for i in range(8):
    class_count = np.load("image_segmentation/mask_count/count_masks_train_"+str(i*100)+".npy")
    if (i == 0):
        counts = class_count
    else:
        counts = np.concatenate((counts, class_count))
    
np.save("image_segmentation/mask_count/maskcount_combined800", counts)


