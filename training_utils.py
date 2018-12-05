#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:42:26 2018

@author: ryangreen
"""

import time
#import numpy as np
import tensorflow as tf
#from skimage import color
import os
from os.path import join

path_of_script = os.path.dirname(os.path.abspath(__file__))
dir_root = join(path_of_script, 'vivid_past_model_files')


def maybe_create_folder(folder):
    os.makedirs(folder, exist_ok=True)


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square erros
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


def training_pipeline(colorizer, l_channel, ab_true, learning_rate, batch_size, num_samples):
    
    predicted_ab = colorizer.build(l_channel)
#    cost = tf.reduce_mean(tf.squared_difference(predicted_ab, ab_true), name="mse")
#    summary = tf.summary.scalar('cost training', cost)
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


def evaluation_pipeline(colorizer, l_channel, ab_channel):
    # Set up validation (input queues, graph)
    imgs_ab_val = colorizer.build(l_channel)
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


def metrics_system(run_id, sess):
    # Merge all the summaries and set up the writers
    dir_metrics = join(dir_root, run_id, 'metrics')
    maybe_create_folder(dir_metrics)
    
    
    train_writer = tf.summary.FileWriter(dir_metrics, sess.graph)
    return train_writer
#
#
def checkpointing_system(run_id):
    # Add ops to save and restore all the variables.
    run_path = join(dir_root, run_id)
    checkpoint_path = join(run_path, 'checkpoints')
    maybe_create_folder(checkpoint_path)
    
    saver = tf.train.Saver(max_to_keep=3)
    latest_checkpoint = tf.train.latest_checkpoint(run_path)
    return  saver, checkpoint_path, latest_checkpoint

#
#def plot_evaluation(res, run_id, epoch):
#    maybe_create_folder(join(dir_root, 'images', run_id))
#    for k in range(len(res['imgs_l'])):
#        img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
#        img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
#                                res['imgs_ab'][k])
#        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
#                              res['imgs_true_ab'][k])
#        top_5 = np.argsort(res['imgs_emb'][k])[-5:]
#        try:
#            top_5 = ' / '.join(labels_to_categories[i] for i in top_5)
#        except:
#            top_5 = str(top_5)
#
#        plt.subplot(1, 3, 1)
#        plt.imshow(img_gray)
#        plt.title('Input (grayscale)')
#        plt.axis('off')
#        plt.subplot(1, 3, 2)
#        plt.imshow(img_output)
#        plt.title('Network output')
#        plt.axis('off')
#        plt.subplot(1, 3, 3)
#        plt.imshow(img_true)
#        plt.title('Target (original)')
#        plt.axis('off')
#        plt.suptitle(top_5, fontsize=7)
#
#        plt.savefig(join(
#            dir_root, 'images', run_id, '{}_{}.png'.format(epoch, k)))
#        plt.clf()
#        plt.close()
#
#
#def l_to_rgb(img_l):
#    """
#    Convert a numpy array (l channel) into an rgb image
#    :param img_l:
#    :return:
#    """
#    lab = np.squeeze(255 * (img_l + 1) / 2)
#    return color.gray2rgb(lab) / 255
#
#
#def lab_to_rgb(img_l, img_ab):
#    """
#    Convert a pair of numpy arrays (l channel and ab channels) into an rgb image
#    :param img_l:
#    :return:
#    """
#    lab = np.empty([*img_l.shape[0:2], 3])
#    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
#    lab[:, :, 1:] = img_ab * 127
#    return color.lab2rgb(lab)