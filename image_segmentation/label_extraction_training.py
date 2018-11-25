#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:25:42 2018

@author: justinelo
"""
from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3
deeplab_model = Deeplabv3()
#img = plt.imread("imgs/image1.jpg")


L_images = np.load("gray_scale.npy")
[count, r, c] = L_images.shape

iteration = 100
batch = 100

for i in range(9, iteration):
    print("ITERATION " + str(i))
    img_L = L_images[i*batch, :, :]
    img = np.stack((img_L, img_L, img_L), axis = 2)
    w, h, _ = img.shape
    ratio = 512. / np.max([w,h])
    resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
    resized = resized / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
    res = deeplab_model.predict(np.expand_dims(resized2,0))
    labels = np.argmax(res.squeeze(),-1)
    #plt.imshow(labels)
    
    #Count number of objects (i.e. count number of different masks)
    set_flat = set(labels.flatten())
    count_masks_train = np.array([len(set_flat)])
    
    results_train = res
    #label_results_train = np.array([labels])

    
    for j in range(i*batch + 1, i*batch+batch):
        print("RANGE: " + str(j))
        img_L = L_images[j, :, :]
        img = np.stack((img_L, img_L, img_L), axis = 2)
        w, h, _ = img.shape
        ratio = 512. / np.max([w,h])
        resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
        resized = resized / 127.5 - 1.
        pad_x = int(512 - resized.shape[0])
        resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
        res = deeplab_model.predict(np.expand_dims(resized2,0))
        labels = np.argmax(res.squeeze(),-1)
        #plt.figure(i)
        #plt.imshow(labels)
        
        
        set_flat = set(labels.flatten())
        
        count_masks_train = np.concatenate((count_masks_train, np.array([len(set_flat)])), axis = 0)
        results_train = np.concatenate((results_train, res), axis = 0)
        #label_results_train = np.concatenate((label_results_train, np.array([labels])), axis = 0)
    
    
    np.save("results_train_" + str(i*batch) + ".npy", results_train)
    np.save("count_masks_train_" + str(i*batch) + ".npy", count_masks_train)
    #np.save("label_results_train_" + str(i*batch) + ".npy", label_results_train)
    
    
    
    
    
    
    
    
    
    
    
    
##########################################    
    
#img_L = L_images[0, :, :]
#img = np.stack((img_L, img_L, img_L), axis = 2)
#w, h, _ = img.shape
#ratio = 512. / np.max([w,h])
#resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
#resized = resized / 127.5 - 1.
#pad_x = int(512 - resized.shape[0])
#resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
#res = deeplab_model.predict(np.expand_dims(resized2,0))
#labels = np.argmax(res.squeeze(),-1)
##plt.imshow(labels)
#
##Count number of objects (i.e. count number of different masks)
#set_flat = set(labels.flatten())
#count_masks_train = np.array([len(set_flat)])
#
#results_train = res
#label_results_train = np.array([labels])
#
#
#
#for i in range(1, 500):
#    img_L = L_images[i, :, :]
#    img = np.stack((img_L, img_L, img_L), axis = 2)
#    w, h, _ = img.shape
#    ratio = 512. / np.max([w,h])
#    resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
#    resized = resized / 127.5 - 1.
#    pad_x = int(512 - resized.shape[0])
#    resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
#    res = deeplab_model.predict(np.expand_dims(resized2,0))
#    labels = np.argmax(res.squeeze(),-1)
#    #plt.figure(i)
#    #plt.imshow(labels)
#    
#    
#    set_flat = set(labels.flatten())
#    
#    count_masks_train = np.concatenate((count_masks_train, np.array([len(set_flat)])), axis = 0)
#    results_train = np.concatenate((results_train, res), axis = 0)
#    label_results_train = np.concatenate((label_results_train, np.array([labels])), axis = 0)
#    
#
#np.save("results_train.npy", results_train)
#np.save("count_masks_train.npy", count_masks_train)
#np.save("label_results_train.npy", label_results_train)


#plt.imshow(labels[:-pad_x])


#img_o = plt.imread("imgs/image1.jpg")
#plt.imshow(img_o)


#plt.imshow([[3, 3, 7, 3, 3], [9 ,10 ,9 ,9, 9],[3, 3, 3, 3, 3], [4, 4, 7, 4, 4]])