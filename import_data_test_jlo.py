#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:37:52 2018
@author: ryangreen & justinelo
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2
#from skimage import lab2rgb

L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:10, :, :]
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:10, :, :]
pred_AB = np.round(np.load("10_AB_predictions.npy", mmap_mode='r') * 255).astype('uint8')


#def scale_LAB(array):
#    assert len(array.shape) == 3
#    for x in range(array.shape[0]):
#        for y in range(array.shape[1]):
#             array[x,y] -= np.array([0, 128, 128]).astype('uint8')
#             array[x,y,0] = (array[x,y,0] * (100/255)).astype('uint8')
#    return array


# Synthesize and save ground truth vs predicted values of numpy arrays
#   4D color arrays should be in shape (num_samples, width, height, channels)
#   3D grayscale array should be in shape (num_samples, width, height)
#   prints according to RGB values -- must be converted
def comparePredictions(L_channel, AB_channel, pred_AB, plot=False):
    assert AB_channel.shape == pred_AB.shape
    assert L_channel.shape[0] == AB_channel.shape[0]
    num_samples = L_channel.shape[0]
    
    for index in range(num_samples):
        synthesized_img_array = np.stack([L_channel[index,:,:], 
                                          pred_AB[index,:,:,0],
                                          pred_AB[index,:,:,1]], 
                                            axis=2)
        misc.imsave('guess' + str(index) + '.png', synthesized_img_array)
        if (plot):
#            plt.imshow(synthesized_img_array)
#            plt.axis('off')
#            plt.title('Predicted Img #' + str(index))
#            plt.show()
            display_LAB_img(synthesized_img_array, 'Predict', index)

        synthesized_img_array = np.stack([L_channel[index,:,:], 
                                          AB_channel[index,:,:,0], 
                                          AB_channel[index,:,:,1]], axis=2)
        misc.imsave('truth' + str(index) + '.png', synthesized_img_array)
        if (plot):
#            plt.imshow(synthesized_img_array)
#            plt.axis('off')
#            plt.title('Truth Img #' + str(index))
#            plt.show()
            display_LAB_img(synthesized_img_array, 'Truth', index)
            
def display_LAB_img(img_lab, title_str, index):
    #convert to RGB
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB) 
  
    # plot the image 
    plt.imshow(img_rgb, interpolation='nearest')
    plt.axis('off')
    plt.title( title_str + ' Img #' + str(index))
    plt.show()