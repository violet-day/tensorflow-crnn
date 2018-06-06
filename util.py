# -*- coding: utf-8 -*-

import time
import numpy as np
from tensorflow.python.platform import flags

seed = np.random.RandomState(int(round(time.time())))

pool_size = 16


def define_flags():
    flags.DEFINE_integer('image_height', 64, 'image height')
    flags.DEFINE_integer('image_width', 224, 'image width')
    flags.DEFINE_integer('image_channel', 3, 'image channel')

    flags.DEFINE_integer('batch_size', 40, 'batch size')

    flags.DEFINE_integer('min_len', 5, 'min len')
    flags.DEFINE_integer('max_len', 10, 'max len')

    flags.DEFINE_integer('n_classes', 37, 'classes num')

    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('learning_decay_steps', 3000, 'learning decay steps')
    flags.DEFINE_float('learning_decay_rate', 0.95, 'learning decay rate')
