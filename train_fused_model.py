


"""
Created on Fri Nov 23 18:50:04 2018

@author: ryangreen
"""

import numpy as np
import os
from os.path import join
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from vivid_past_model_definition import VividPastAutoEncoder

from training_utils import fused_training_pipeline, checkpointing_system, fused_evaluation_pipeline, maybe_create_folder
#import import_data_test_jlo
    

    
# PARAMETERS
run_id = 'fusion_3'
total_epochs = 2000
num_test_samples = 50
total_train_samples = 1000
batch_size = 20
learning_rate = 0.002
save_freq = 10

# calculated params
batches = total_train_samples // batch_size

root_dir = join('vivid_past_model_files', run_id)
metrics_dir = join(root_dir, 'metrics')
predictions_dir = join(root_dir, 'predictions') 

low_loss = -1;

print('~~~~~~~~~~~~TRAINING ' + run_id + '~~~~~~~~~~~~')
print('training on ' + str(total_train_samples) + ' images')
print('testing on ' + str(num_test_samples) + ' images')
print('batch size: ' + str(batch_size))
print('set to save every: ' + str(save_freq) + ' epochs\n')

epoch_cost_save_path = metrics_dir +'/epoch_costs'
maybe_create_folder(metrics_dir)
maybe_create_folder(predictions_dir)

# load file tracking loss history
if os.path.isfile(epoch_cost_save_path + '.npy'):
    print("Found metrics file:")
    epoch_costs = np.load(epoch_cost_save_path + '.npy').item()
    # reset lowest validation loss
    low_loss = np.min(epoch_costs['validation'])
    print('Low loss: ' + str(low_loss))
    epoch_id = len(epoch_costs['training']) + 1
    print('Last epoch finished: ', epoch_id - 1)
else:
    print('No metrics file found')
    epoch_costs = {'training': np.array([]), 'validation': np.array([])}
    epoch_id = 1

print('\n')

##### LOAD DATA ####

def load_data():
    L_channel = np.load("image-colorization/gray_scale.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    L_channel = np.expand_dims(L_channel, axis=3)
    L_channel = ((L_channel - np.mean(L_channel, axis=(0,1,2))) / np.std(L_channel, axis=(0,1,2)) / 2).astype('float32')
    AB_channel = np.load("image-colorization/ab/ab1.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    AB_channel = ((AB_channel - np.mean(AB_channel, axis=(0,1,2))) / np.std(AB_channel, axis=(0,1,2)) / 2).astype('float32')
    seg_data_raw = np.load("segmentation_arrays/segmentation_data_5000.npy", mmap_mode='r')[:total_train_samples+num_test_samples, :, :]
    seg_data = seg_data_raw / np.max(seg_data_raw, axis=(0,1,2))

    # Create tf Dataset and iterator for feeding data in batches
    training_data = tf.data.Dataset.from_tensor_slices((L_channel[num_test_samples:], AB_channel[num_test_samples:],seg_data[num_test_samples:]))
    training_data = training_data.repeat().shuffle(buffer_size=50).batch(batch_size)
    iterator = training_data.make_one_shot_iterator()
    
    # make Validation Dataset
    testing_data_L = L_channel[:num_test_samples,:,:]
    testing_data_AB = AB_channel[:num_test_samples,:,:]
    testing_seg = seg_data[:num_test_samples,:,:]
    
    return training_data, iterator, testing_data_L, testing_data_AB, testing_seg

##### LOAD DATA ####

training_data, iterator, testing_data_L, testing_data_AB, testing_seg = load_data()
###### BUILD MODEL #######

# START
#saver = tf.train.Saver()

sess = tf.Session()
#K.set_session(sess)

# Build the network and the various operations
colorizer = VividPastAutoEncoder(149)

# get next batch
l_channel, ab_true, seg = iterator.get_next()
training = fused_training_pipeline(colorizer, l_channel, ab_true, seg, learning_rate, batch_size, total_train_samples)
evaluation = fused_evaluation_pipeline(colorizer, testing_data_L, testing_seg, testing_data_AB)
#summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)

#low_loss = tf.Variable(-1, name='low_loss', trainable=False)
#epoch_id = tf.Variable(1, name='epoch_id', trainable=False)

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore
    if latest_checkpoint is not None:
        print('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print(' done!', run_id)
    else:
        print('No checkpoint found in: {}'.format(checkpoint_paths), run_id)
        


    while epoch_id <= total_epochs:
        print('Starting epoch: {} (total images {})'
                  .format(epoch_id, total_train_samples))
        # Training step
        epoch_training_loss = 0
        for batch in range(batches):
            print('Batch: {}/{}'.format(batch+1, batches))
            res = sess.run(training)
            global_step = res['global_step']
            print('Cost: {} Global step: {}'.format(res['cost'], global_step))
            epoch_training_loss += res['cost']
            
            if (batch == 0 and epoch_id % save_freq == 0):
                np.save(predictions_dir + '/training_pred_ab_ep' + str(epoch_id), res['training_pred'])
#            summary_writer.add_summary(res['summary'], global_step)
        epoch_training_loss /= batches
        
        print()
        print('Epoch {} Ended. Validating...'.format(epoch_id))
        # Evaluation step on validation
        res = sess.run(evaluation)
        epoch_val_cost = res['cost']
        # add costs to arrays
        epoch_costs['training'] = np.append(epoch_costs['training'], epoch_training_loss)
        epoch_costs['validation'] = np.append(epoch_costs['validation'], epoch_val_cost)
        np.save(epoch_cost_save_path, epoch_costs)
        
#        print(sess.run(epoch_id))
        print('Training loss: {}'.format(epoch_training_loss))
        print('Validation loss: {}'.format(epoch_val_cost))
        
        if (low_loss < 0 or low_loss > epoch_val_cost or epoch_id % save_freq == 0):
#            lowest_validation
            if (low_loss < 0 or low_loss > epoch_val_cost):
                low_loss = epoch_val_cost
                print('Improved Loss!!')
            print('Saving model...')
            # Save the variables to disk
            save_path = saver.save(sess, checkpoint_paths, global_step)
            
#            import_data_test_jlo.comparePredictions(testing_data_L, testing_data_AB, res['predicted_ab']):
            print("Model saved in: %s" % save_path, run_id)
            
        if (epoch_id % save_freq == 0):
            # Save predictions for validation set
            print('Saving predictions for epoch ', str(epoch_id))
            np.save(predictions_dir + '/pred_ab' + '_ep' + str(epoch_id) + '_'+ str(epoch_val_cost), res['predicted_ab'])
        
        # increment epoch
        epoch_id = epoch_id + 1
        print('----------------------------------------')



