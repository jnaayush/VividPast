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

from vivid_past_model_definition import VividPastAutoEncoder

from training_utils import training_pipeline, checkpointing_system, evaluation_pipeline
#        plot_evaluation, training_pipeline, metrics_system, print_log
    
    
# PARAMETERS
run_id = 'toyrun1'
epochs = 2
num_test_samples = 10
total_train_samples = 10
batch_size = 10
learning_rate = 0.001
batches = total_train_samples // batch_size


##### LOAD DATA ####

L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:total_train_samples, :, :]
L_channel = (L_channel / 255).astype('float32')
L_channel = np.expand_dims(L_channel, axis=3)
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:total_train_samples, :, :]
AB_channel = (AB_channel / 255).astype('float32')

# Create tf Dataset and iterator for feeding data in batches
training_data = tf.data.Dataset.from_tensor_slices((L_channel, AB_channel))
training_data = training_data.repeat().shuffle(buffer_size=50).batch(batch_size)
iterator = training_data.make_one_shot_iterator()

###### BUILD MODEL #######

# START
#saver = tf.train.Saver()

sess = tf.Session()
#K.set_session(sess)

# Build the network and the various operations
colorizer = VividPastAutoEncoder()

# get next batch
l_channel, ab_true = iterator.get_next()
training = training_pipeline(colorizer, l_channel, ab_true, learning_rate, batch_size, total_train_samples)
evaluation = evaluation_pipeline(colorizer, l_channel, ab_true, num_test_samples)
#summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
#    lowest_validation_loss = tf.Variable(-1, name='lowest_validation_loss')
    low_loss = -1;

    # Coordinate the loading of image files.
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)

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
            res = sess.run(training)
            global_step = res['global_step']
            print('Cost: {} Global step: {}'
                      .format(res['cost'], global_step))
#            summary_writer.add_summary(res['summary'], global_step)

        # Evaluation step on validation
        res = sess.run(evaluation)
        epoch_cost = res['cost']
        print()
        print('Epoch {} Ended. Validating...'.format(epoch))
        print('Validation loss: {}'.format(epoch_cost))
        if (low_loss < 0 or low_loss > epoch_cost):
#            lowest_validation
            print('Improved Loss. Saving model...')
            # Save the variables to disk
#            save_path = saver.save(sess, checkpoint_paths)
#            print("Model saved in: %s" % save_path)
            
            save_path = saver.save(sess, checkpoint_paths, global_step)
            print("Model saved in: %s" % save_path, run_id)
        print('----------------------------------------')



    # Finish off the filename queue coordinator.
#    coord.request_stop()
#    coord.join(threads)