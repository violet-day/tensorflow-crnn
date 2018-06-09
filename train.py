# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib import slim
from util import define_flags, pool_size

define_flags()

from model import foward, create_train_op
from provider import get_input


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    images, image_widths, labels = get_input()
    seq_len_inputs = tf.divide(image_widths, pool_size, name='seq_len_input_op') - 1

    logprob = foward(images, is_training=True)
    train_op = create_train_op(labels, seq_len_inputs, logprob)

    slim.learning.train(
        train_op=train_op,
        logdir=flags.FLAGS.logdir,
        number_of_steps=100000,
        save_summaries_secs=60,
        save_interval_secs=120)


if __name__ == '__main__':
    flags.DEFINE_string('logdir', 'logs/train', 'log dir')
    tf.app.run()
