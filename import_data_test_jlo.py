#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:37:52 2018
@author: justinelo
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2
import re
#from skimage import lab2rgb

from training_utils import maybe_create_folder

def parseEpoch(filename):
    pattern = re.compile('ep\d+')
    return pattern.search(filename).group()

run_id = 'no_fusion_3'
folder = 'test_predictions/' + run_id + '/'
filename = 'training_pred_ab_ep100.npy'
epoch = parseEpoch(filename)
num_imgs = 50
start_index = 0

dataset_type = '/train/'
#dataset_type = '/validation/'

save_path = 'visualize/' + run_id + dataset_type + epoch + '/'

# load pre-computed scaling factors
factors = np.load('scaling-factors-all-data.npy').item()

# for training prediction
L_channel = np.round((np.load(folder + "l_channel_" + epoch + ".npy", mmap_mode='r') * factors["L_factor"]) + factors["L_mu"]).astype('uint8')[:,:,:,0]
AB_channel = np.round((np.load(folder + "ab_channel_" + epoch + ".npy", mmap_mode='r') * factors["AB_factor"]) + factors["AB_mu"]).astype('uint8')
pred_AB = np.round((np.load(folder + filename, mmap_mode='r') * factors["AB_factor"]) + factors["AB_mu"]).astype('uint8')

# for validation predictions
#L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[start_index:num_imgs+start_index, :, :]
#AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[start_index:num_imgs+start_index, :, :]
#pred_AB = np.round((np.load(folder + filename, mmap_mode='r') * factors["AB_factor"]) + factors["AB_mu"]).astype('uint8')


# Synthesize and save ground truth vs predicted values of numpy arrays
#   4D color arrays should be in shape (num_samples, width, height, channels)
#   3D grayscale array should be in shape (num_samples, width, height)
#   prints according to RGB values -- must be converted
def comparePredictions(L_channel, AB_channel, pred_AB, plot=False, savePred=False, saveTruth=False):
    assert AB_channel.shape == pred_AB.shape
    assert L_channel.shape[0] == AB_channel.shape[0]
    num_samples = L_channel.shape[0]
    
    maybe_create_folder(save_path)
    
    for index in range(num_samples):
        synthesized_img_array = np.stack([L_channel[index,:,:], 
                                          pred_AB[index,:,:,0],
                                          pred_AB[index,:,:,1]], 
                                            axis=2)

        rgb = display_LAB_img(synthesized_img_array, 'Predict', index, plot=plot)
        if (savePred):
            misc.imsave(save_path + 'guess' + str(index) + '.png', rgb)

        synthesized_img_array = np.stack([L_channel[index,:,:], 
                                          AB_channel[index,:,:,0], 
                                          AB_channel[index,:,:,1]], axis=2)

        rgb = display_LAB_img(synthesized_img_array, 'Truth', index, plot=plot)
        if (saveTruth):
            misc.imsave(save_path + 'truth' + str(index) + '.png', rgb)
            
def display_LAB_img(img_lab, title_str, index, plot=True):
    #convert to RGB
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB) 
  
    # plot the image 
    if plot:
        plt.imshow(img_rgb, interpolation='nearest')
        plt.axis('off')
        plt.title( title_str + ' Img #' + str(index))
        plt.show()
    return img_rgb
    

comparePredictions(L_channel, AB_channel, pred_AB, True, True, True)