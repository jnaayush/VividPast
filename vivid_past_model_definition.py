#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:51:14 2018

@author: ryangreen
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, concatenate


#####################################################################

class VividPastAutoEncoder():
    
    def __init__(self, fused_layer_length):
        self.encoder = build_encoder()
        self.fusion_layer = Conv2D(fused_layer_length, (1, 1), activation='relu')
        self.decoder = build_decoder(fused_layer_length)

    def build(self, l_channel):
        img_enc = self.encoder(l_channel)
        return self.decoder(img_enc)
    
    def build_with_fusion(self, l_channel, seg_data):
        img_enc = self.encoder(l_channel)
        post_fusion = self.fusion_layer(fuse(img_enc, seg_data))
        return self.decoder(post_fusion)
#####################################################################

def fuse( encoded_output, classification_output):
    return concatenate([encoded_output, classification_output], axis=3)
        # CONCAT image semantic segmentation output with encoder output layer
  
#####################################################################  
################### Build functions #################################
        
def build_encoder():
    model = tf.keras.Sequential(name='encoder')
#    model.add(InputLayer(input_shape=(None,None,1), dtype='float32'))
#    # Input: 224x224x1
#    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
#    # Now: 112x112x64
#    model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=1))
#    # Now: 112x112x64
#    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
#    # Now: 56x56x128
#    model.add(Conv2D(128, (2, 2), activation='relu', padding='same', strides=1))
#    # Now: 56x56x128
#    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
#    # Now: 28x28x128
    
    model.add(InputLayer(input_shape=(None,None,1), dtype='float32'))
    # Input: 224x224x1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # Now: 112x112x32
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same', strides=1))
    # Now: 112x112x32
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    # Now: 56x56x64
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=1))
    # Now: 56x56x64
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    # Now: 28x28x128
    
    
#    print("USING TEST ARCHITECTURE -- change network def file")
#    model.add(Conv2D(64, (4, 4), activation='relu', padding='same', strides=4))
    
    return model

def build_decoder(fused_layer_length):
    model = tf.keras.Sequential(name='decoder')
    model.add(InputLayer(input_shape=(None, None, fused_layer_length), dtype='float32'))
    # Input: 28x28x256
    model.add(UpSampling2D((2, 2)))
    # Now 56x56x256
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # Now 56x56x128
    model.add(UpSampling2D((2, 2)))
    # Now 112x112x128
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # Now 112x112x64
    model.add(UpSampling2D((2, 2)))
    # Now 224x224x64
    model.add(Conv2D(2, (1, 1), activation='tanh', padding='same'))
    # Now 224x224x2
    return model