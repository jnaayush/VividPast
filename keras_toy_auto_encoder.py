#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:39:55 2018

@author: ryangreen
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
#import matplotlib.pyplot as plt
#from scipy import misc

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = tf.keras.Sequential()
model.add(InputLayer(input_shape=(224,224,1), dtype='float32'))
# Input: 224x224x1
model.add(Conv2D(32, (2, 2), activation='relu', padding='same', strides=2))
# Now: 112x112x32
model.add(Conv2D(64, (4, 4), activation='relu', padding='same', strides=4))
# Now: 28x28x64

# Decode
model.add(UpSampling2D((2, 2)))
# Now 56x56x64
model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
# Now 56x56x32
model.add(UpSampling2D((2, 2)))
# Now 112x112x32
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Now 112x112x32
model.add(UpSampling2D((2, 2)))
# Now 224x224x32
model.add(Conv2D(2, (1, 1), activation='relu', padding='same'))
# Now 224x224x2



## compile model
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error


tf.logging.set_verbosity(tf.logging.INFO)

epoch_num = 3
batch_size = 20
num_samples = 500

#### DATA ####
L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:num_samples, :, :]
L_channel = (L_channel / 255).astype('float32')
L_channel = np.expand_dims(L_channel, axis=3)
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:num_samples, :, :]
AB_channel = (AB_channel / 255).astype('float32')

training_data = tf.data.Dataset.from_tensor_slices((L_channel, AB_channel))
training_data = training_data.batch(20).repeat()


model.fit(training_data, epochs=epoch_num, steps_per_epoch=20)
## NOTE: add validation data

#print("PREDICTING")
pred_AB = model.predict(L_channel[:10])

np.save('10_AB_predictions', pred_AB)


#for index in range(5):
#    synthesized_img_array = np.stack([L_channel[index,:,:,0], pred_AB[index,:,:,0],pred_AB[index,:,:,1]], axis=2)
#    misc.imsave('test' + str(index) + '.png', synthesized_img_array)
#    plt.imshow(synthesized_img_array)
#    plt.axis('off')
#    plt.show()
