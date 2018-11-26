#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:50:04 2018

@author: ryangreen
"""

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from vivid_past_model_definition_with_seg import VividPastAutoEncoderWithSeg
from save_config import maybe_create_folder
from fusion_training_utils import training_pipeline, checkpointing_system, evaluation_pipeline, savePredictions
#import import_data_test_jlo
    
    
# PARAMETERS
run_id = 'fusion_train_1'
epochs = 1000
num_test_samples = 100
total_train_samples = 1000
batch_size = 10
learning_rate = 0.001
batches = total_train_samples // batch_size


###### BUILD MODEL #######


sess = tf.Session()

# Initialize colorizer model
colorizer = VividPastAutoEncoderWithSeg(277)


##### LOAD DATA ####

L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:total_train_samples, :, :]
L_channel = ((L_channel / 127.5) -1).astype('float32')
L_channel = np.expand_dims(L_channel, axis=3)
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:total_train_samples, :, :]
AB_channel = ((AB_channel / 127.5) -1).astype('float32')
seg_data = np.load("segmentation_arrays/seg_data_downsampled_1000.npy", mmap_mode='r')[:total_train_samples, :, :]
epoch_cost_array = []



# Create tf Dataset and iterator for feeding data in batches
training_data = tf.data.Dataset.from_tensor_slices((L_channel, AB_channel, seg_data))
training_data = training_data.repeat().shuffle(buffer_size=50).batch(batch_size)
iterator = training_data.make_one_shot_iterator()

# make Validation Dataset
testing_data_L = L_channel[:num_test_samples,:,:]
testing_data_AB = AB_channel[:num_test_samples,:,:]
testing_seg_data = seg_data[:num_test_samples,:,:]

## get first batch
l_channel, ab_true , seg = iterator.get_next()
training = training_pipeline(colorizer, l_channel, ab_true, seg, learning_rate, batch_size)
evaluation = evaluation_pipeline(colorizer, testing_data_L, testing_data_AB, testing_seg_data)
#summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)

epoch_cost_array = []

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
#    lowest_validation_loss = tf.Variable(-1, name='lowest_validation_loss')
    low_loss = -1;


    # Restore
    if latest_checkpoint is not None:
        print('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print(' done!', run_id)
    else:
        print('No checkpoint found in: {}'.format(checkpoint_paths), run_id)

    for epoch in range(epochs):
        print('Starting epoch: {} (total images {})'
                  .format(epoch+1, total_train_samples))
        # Training step
        for batch in range(batches):
            print('Batch: {}/{}'.format(batch+1, batches))
            # get next batch
            res = sess.run(training)
            global_step = res['global_step']
            print('Cost: {} Global step: {}'
                      .format(res['cost'], global_step))
#            summary_writer.add_summary(res['summary'], global_step)

        # Evaluation step on validation
        res = sess.run(evaluation)
        epoch_cost = res['cost']
        print()
        print('Epoch {} Ended. Validating...'.format(epoch+1))
        print('Validation loss: {}'.format(epoch_cost))
        if (low_loss < 0 or low_loss > epoch_cost):
#            lowest_validation
            print('Improved Loss. Saving model...')
            # Save the variables to disk
            save_path = saver.save(sess, checkpoint_paths)
            # Save predictions for validation set
            savePredictions(run_id, global_step, res['predicted_ab'])
#            import_data_test_jlo.comparePredictions(testing_data_L, testing_data_AB, res['predicted_ab']):
            print("Model saved in: %s" % save_path, run_id)
        print('----------------------------------------')
        save_dir = os.path.join('epoch_cost_history', run_id)
        maybe_create_folder(save_dir)
        epoch_cost_array.append(epoch_cost)
        np.save(save_dir+'/cost_history.npy', epoch_cost_array)