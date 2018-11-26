
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Fri Nov 23 17:51:14 2018
    
    @author: ryangreen
    """
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, concatenate, Lambda, MaxPooling2D
from keras.backend import tf as ktf



#####################################################################

class VividPastAutoEncoderWithSeg():
    
    def __init__(self, fused_layer_length):
        self.encoder = build_encoder()
        #        self.fusion = FusionLayer()
        #        self.after_fusion = Conv2D(depth_after_fusion, (1, 1), activation='relu')
        self.decoder = build_decoder(fused_layer_length)
    
    def build(self, l_channel, seg_data):
        img_enc = self.encoder(l_channel)
        fusion = fuse(img_enc, seg_data)
        #        fusion = self.after_fusion(fusion)
        
        return self.decoder(fusion)

#####################################################################

    
def fuse( encoded_output, classification_output):
#    model = tf.keras.Sequential(name='fusion')
#    model.add(InputLayer(input_shape=(None,None,1), dtype='float32'))
#    # ten = tf.convert_to_tensor(x, dtype=tf.float32)
#    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_last'))
#    # 256 x 256
#    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_last'))
#    #128 x 128
#    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_last'))
#    #64 x 64
#    model.add(Lambda(lambda x: ktf.image.resize_images(x,(56, 56))))
#    #56 x 56
#    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_last'))
#    #28x28
#    
    return concatenate([encoded_output, classification_output], axis=3)
# CONCAT image semantic segmentation output with encoder output layer

#####################################################################
################### Build functions #################################

def build_encoder():
    model = tf.keras.Sequential(name='encoder')
    model.add(InputLayer(input_shape=(None,None,1), dtype='float32'))
    # Input: 224x224x1
    #    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    #    # Now: 112x112x64
    #    model.add(Conv2D(128, (2, 2), activation='tanh', padding='same', strides=1))
    #    # Now: 112x112x128
    #    model.add(Conv2D(128, (3, 3), activation='tanh', padding='same', strides=2))
    #    # Now: 64x64x128
    #    model.add(Conv2D(256, (2, 2), activation='tanh', padding='same', strides=1))
    #    # Now: 64x64x256
    #    model.add(Conv2D(256, (3, 3), activation='tanh', padding='same', strides=2))
    #    # Now: 32x32x256
    

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    
    
    #    # for testing -- REMOVE
    #    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=4))
    #    # Now: 28x28x64
    return model

def build_decoder(fused_layer_length):
    model = tf.keras.Sequential(name='decoder')
    model.add(InputLayer(input_shape=(None, None, fused_layer_length), dtype='float32'))
    # Input: 28x28x256
    #    model.add(UpSampling2D((2, 2)))
    #    # Now 56x56x256
    #    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #    # Now 56x56x128
    #    model.add(UpSampling2D((2, 2)))
    #    # Now 112x112x128
    #    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #    # Now 112x112x64
    #    model.add(UpSampling2D((2, 2)))
    #    # Now 224x224x64
    #    model.add(Conv2D(2, (1, 1), activation='sigmoid', padding='same'))
    #     Now 224x224x2
    
    model.add(UpSampling2D((2, 2)))
    # 56
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    # 112
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    # 224
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    return model