#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baldassareFe
"""

import itertools
import os
from os.path import expanduser, join

path_of_script = os.path.dirname(os.path.abspath(__file__))

# Default folders
dir_root = join(expanduser('~'), 'vivid_past_model_files')
dir_originals = join(dir_root, 'original')
dir_resized = join(dir_root, 'resized')
dir_tfrecord = join(dir_root, 'tfrecords')
dir_metrics = join(dir_root, 'metrics')
dir_checkpoints = join(dir_root, 'checkpoints')


def maybe_create_folder(folder):
    os.makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)