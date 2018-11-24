#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baldassareFe
"""

import itertools
from os import makedirs
from os.path import expanduser, join

# Default folders
dir_root = join(expanduser('~'), 'vivid_past_model_files')
dir_originals = join(dir_root, 'original')
dir_resized = join(dir_root, 'resized')
dir_tfrecord = join(dir_root, 'tfrecords')
dir_metrics = join(dir_root, 'metrics')
dir_checkpoints = join(dir_root, 'checkpoints')


def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)