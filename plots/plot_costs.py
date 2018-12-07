#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:34:00 2018

@author: ryangreen
"""

import numpy as np
import matplotlib.pyplot as plt

fusion_file = 'fusion_3_epoch_costs.npy'
non_fusion_file = 'no_fusion_3_epoch_costs_107.npy'

start_ep = 0
end_ep = 107

def loadData(fusion_file, non_fusion_file):
    fusion_costs = np.load(fusion_file).item()
    non_fusion_costs = np.load(non_fusion_file).item()
    f_train = fusion_costs['training']
    f_test = fusion_costs['validation']
    nf_train = non_fusion_costs['training']
    nf_test = non_fusion_costs['validation']
    return f_train, f_test, nf_train, nf_test

f_train, f_validation, nf_train, nf_validation = loadData(fusion_file, non_fusion_file)
#print('Found ' + str(train.shape[0]) + ' data points.\n')

plt.figure()
plt.plot(f_train[start_ep:end_ep], label='Training (Fusion)')
plt.plot(f_validation[start_ep:end_ep], label='Validation (Fusion)')
plt.plot(nf_train[start_ep:end_ep], label='Training (No Fusion)')
plt.plot(nf_validation[start_ep:end_ep], label='Validation (No Fusion)')
plt.legend()
plt.xlabel('Epoch number')
plt.ylabel('Loss')
#plt.yscale('log')
#plt.show()
plt.savefig('training_costs_plot.png')

#plt.plot(validation[2:])
#plt.show()
