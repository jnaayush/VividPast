#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:42:26 2018

@author: ryangreen
"""

import time
from os.path import join
from numpy import save

from save_config import dir_checkpoints, dir_root, maybe_create_folder
#import numpy as np
import tensorflow as tf
#from skimage import color


#matplotlib.use('Agg')
#matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
#import matplotlib.pyplot as plt

#labels_to_categories = pickle.load(
#    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss function
#    cost = tf.reduce_mean(
#        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    cost = tf.losses.absolute_difference(
        img_ab_true,
        img_ab_out,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cost)
    return cost, summary


def training_pipeline(colorizer, l_channel, ab_true, seg, learning_rate, batch_size):
    
    predicted_ab = colorizer.build(l_channel, seg)

    cost, summary = loss_with_metrics(predicted_ab, ab_true, 'training')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'cost': cost,
        'summary': summary
    }#, irr, read_batched_examples


def evaluation_pipeline(colorizer, l_channel, ab_channel, seg):
    # Set up validation (input queues, graph)
    imgs_ab_val = colorizer.build(l_channel, seg)
    cost, summary = loss_with_metrics(imgs_ab_val, ab_channel, 'validation')
    return {
#        'imgs_l': l_channel,
        'predicted_ab': imgs_ab_val,
#        'imgs_true_ab': ab_channel,
        'cost': cost,
        'summary': summary
    }


def print_log(content, run_id):
    with open('output_{}.txt'.format(run_id), mode='a') as f:
        f.write('[{}] {}\n'.format(time.strftime("%c"), content))


#def metrics_system(run_id, sess):
#    # Merge all the summaries and set up the writers
#    train_writer = tf.summary.FileWriter(join(dir_metrics, run_id), sess.graph)
#    return train_writer
#
#
def checkpointing_system(run_id):
    # Add ops to save and restore all the variables.
    maybe_create_folder(join(dir_root, 'checkpoints', run_id))
    
    saver = tf.train.Saver()
    checkpoint_paths = join(dir_checkpoints, run_id)
    latest_checkpoint = tf.train.latest_checkpoint(dir_checkpoints)
    return  saver, checkpoint_paths, latest_checkpoint

def savePredictions(run_id, epoch, predictions):
    save_dir = join('test_predictions', run_id)
    maybe_create_folder(save_dir)
    return save(save_dir + '/' + 'predicted_ab' + '_ep' + str(epoch+1), predictions)