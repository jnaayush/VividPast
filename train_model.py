#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:50:04 2018

@author: ryangreen
"""

import numpy as np
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from vivid_past_model_definition import VividPastAutoEncoder

from training_utils import training_pipeline, checkpointing_system, evaluation_pipeline
#import import_data_test_jlo
    
    
# PARAMETERS
run_id = 'variable_test'
total_epochs = 30
num_test_samples = 1
total_train_samples = 4
batch_size = 2
learning_rate = 0.001
batches = total_train_samples // batch_size


##### LOAD DATA ####

L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[num_test_samples:total_train_samples+num_test_samples, :, :]
L_channel = (L_channel / 255).astype('float32')
L_channel = np.expand_dims(L_channel, axis=3)
AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[num_test_samples:total_train_samples+num_test_samples, :, :]
AB_channel = (AB_channel / 255).astype('float32')


# Create tf Dataset and iterator for feeding data in batches
training_data = tf.data.Dataset.from_tensor_slices((L_channel, AB_channel))
training_data = training_data.repeat().shuffle(buffer_size=50).batch(batch_size)
iterator = training_data.make_one_shot_iterator()

# make Validation Dataset
testing_data_L = L_channel[:num_test_samples,:,:]
testing_data_AB = AB_channel[:num_test_samples,:,:]

###### BUILD MODEL #######

# START
#saver = tf.train.Saver()

sess = tf.Session()
#K.set_session(sess)

# Build the network and the various operations
colorizer = VividPastAutoEncoder(64)

# get next batch
l_channel, ab_true = iterator.get_next()
training = training_pipeline(colorizer, l_channel, ab_true, learning_rate, batch_size, total_train_samples)
evaluation = evaluation_pipeline(colorizer, testing_data_L, testing_data_AB)
#summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)

#low_loss = tf.Variable(-1, name='low_loss', trainable=False)
epoch_id = tf.Variable(1, name='epoch_id', trainable=False)
low_loss = -1;

with sess.as_default():
#    # Initialize
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())

    # Restore
    if latest_checkpoint is not None:
        print('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print(' done!', run_id)
    else:
        print('No checkpoint found in: {}'.format(checkpoint_paths), run_id)
        
        # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    epochs_not_finished = tf.less(epoch_id,total_epochs)
    not_finished = sess.run(epochs_not_finished)
    while not_finished:
        epoch = sess.run(epoch_id)
        print('Starting epoch: {} (total images {})'
                  .format(epoch, total_train_samples))
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
        epoch_id = epoch_id + 1
#        print(sess.run(epoch_id))
        print('Validation loss: {}'.format(epoch_cost))
        
        if (low_loss < 0 or low_loss > epoch_cost):
#            lowest_validation
            print('Improved Loss. Saving model...')
#            tf.assign(low_loss, epoch_cost)
            # Save the variables to disk
            save_path = saver.save(sess, checkpoint_paths, global_step)
            # Save predictions for validation set
#            np.save('test_predictions' + '_' + run_id, res['predicted_ab'])
#            import_data_test_jlo.comparePredictions(testing_data_L, testing_data_AB, res['predicted_ab']):
            print("Model saved in: %s" % save_path, run_id)
        print('----------------------------------------')
        not_finished = sess.run(epochs_not_finished)

