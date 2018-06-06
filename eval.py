# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib import slim
from util import define_flags, pool_size

define_flags()

from model import foward, create_metrics, create_loss
from provider import get_input


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    images, image_widths, labels = get_input()
    seq_len_inputs = tf.divide(image_widths, pool_size, name='seq_len_input_op')

    logprob = foward(images, is_training=False)

    loss_ctc = create_loss(labels, logprob, seq_len_inputs)
    loss_summary = tf.summary.scalar('tf-crnn/ctc_loss', loss_ctc)

    names_to_values, names_to_updates = create_metrics(logprob, seq_len_inputs, labels)
    summary_ops = []

    for metric_name, metric_value in names_to_values.iteritems():
        op = tf.summary.scalar('tf-crnn/' + metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)
    summary_ops.append(loss_summary)

    tf.train.get_or_create_global_step()

    slim.evaluation.evaluation_loop(
        '',
        flags.FLAGS.checkpoint_dir,
        flags.FLAGS.eval_dir,
        num_evals=10,
        eval_op=names_to_updates.values(),
        summary_op=tf.summary.merge(summary_ops),
        eval_interval_secs=flags.FLAGS.eval_interval_secs)


if __name__ == '__main__':
    flags.DEFINE_integer('eval_interval_secs', 60, 'eval_interval_secs')
    flags.DEFINE_string('checkpoint_dir', 'logs/train', 'checkpoint_dir')
    flags.DEFINE_string('eval_dir', 'logs/eval', 'eval dir')
    tf.app.run()
