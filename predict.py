#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:39:36 2018

@author: ryangreen
"""

import numpy as np
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from vivid_past_model_definition import VividPastAutoEncoder

from training_utils import  evaluation_pipeline, checkpointing_system, maybe_create_folder

factors = np.load('scaling-factors-all-data.npy').item()

# PARAMS
run_id = 'no_fusion_3'
eval_id = 'video_test_1'
gen_dir = 'eval_predictions'
sub_folder = run_id + '/' + eval_id

l_array = None
ab_array = None

sess = tf.Session()
colorizer = VividPastAutoEncoder(128)


def norm_data(l, ab):
    L_channel = ((l  - factors["L_mu"]) / factors["L_factor"] ).astype('float32')
    L_channel = np.expand_dims(L_channel, axis=3)
    AB_channel = ((ab  - factors["AB_mu"]) / factors["AB_factor"] ).astype('float32')
    
    return L_channel, AB_channel

##### LOAD / NORMALIZE DATA ####
l, ab = norm_data(l_array, ab_array)

saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)
evaluation = evaluation_pipeline(colorizer, l, ab)

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
        np.save(save_dir + '/' + 'predicted_ab_' + run_id + '_' + eval_id + '_cost' + str(res['cost']) , res['predicted_ab'])
        print("Predictions saved.")
    
    else:
        print('No model found in: {}'.format(checkpoint_paths), run_id)
        
        



