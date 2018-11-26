#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:39:36 2018

@author: ryangreen
"""

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from vivid_past_model_definition import VividPastAutoEncoder

from training_utils import  evaluation_pipeline, checkpointing_system
from save_config import maybe_create_folder


# PARAMS
run_id = 'no_seg_small1'
gen_dir = 'eval_predictions'
sub_folder = 'test_eval'
num_pred = 10
begin = 0
end = begin + num_pred

sess = tf.Session()
colorizer = VividPastAutoEncoder(256)

##### LOAD DATA ####

L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[begin:end, :, :]
L_channel = ((L_channel / 127.5) -1).astype('float32')
L_channel = np.expand_dims(L_channel, axis=3)
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[begin:end, :, :]
AB_channel = ((AB_channel / 127.5) -1).astype('float32')

saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)
evaluation = evaluation_pipeline(colorizer, L_channel, AB_channel)

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore
    if latest_checkpoint is not None:
        print('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print(' done!', run_id)
        
        res = sess.run(evaluation)
        epoch_cost = res['cost']
        save_dir = os.path.join(gen_dir, sub_folder)
        maybe_create_folder(save_dir)
        np.save(save_dir + '/' + 'predicted_ab_' + str(begin) + '-' + str(end) + '_cost' + str(res['cost']) , res['predicted_ab'])
        print("Predictions saved.")
    
    else:
        print('No model found in: {}'.format(checkpoint_paths), run_id)
        
        



