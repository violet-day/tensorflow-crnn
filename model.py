# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow import logging
from tensorflow.python.platform import flags
from tensorflow.contrib import slim
from tensorflow.contrib.rnn import BasicLSTMCell


def vgg_a(inputs,
          scope='vgg_a', is_training=True):
    batch_norm_params = {
        'is_training': is_training
    }
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                net = slim.repeat(
                    inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5')
                return net


def foward(images, is_training=True):
    tf.summary.image('tf-crnn/images', images)
    dropout_keep_prob = 0.7 if is_training else 1.0

    # with slim.arg_scope(vgg.vgg_arg_scope()):
    cnn_net = vgg_a(images, is_training=is_training)
    logging.info('cnn_net shape: %s' % cnn_net.get_shape())

    # cnn_net = deep_cnn(images, is_training, False)

    with tf.variable_scope('Reshaping_cnn'):
        shape = cnn_net.get_shape().as_list()  # [batch, height, width, features]
        transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features]
        conv_reshaped = tf.reshape(transposed, [shape[0], -1, shape[1] * shape[3]],
                                   name='reshaped')  # [batch, width, height x features]
        logging.info('after reshape cnn, shape: %s' % conv_reshaped.shape)

    list_n_hidden = [256, 256]

    with tf.name_scope('deep_bidirectional_lstm'):
        # Forward direction cells
        fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        conv_reshaped,
                                                                        dtype=tf.float32
                                                                        )
        # Dropout layer
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=dropout_keep_prob)
        logging.info('after lstm shape: %s' % lstm_net.shape)

    with tf.variable_scope('fully_connected'):
        shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
        fc_out = slim.layers.linear(lstm_net, flags.FLAGS.n_classes)  # [batch x width, n_class]
        logging.info('fc_out shape: %s' % fc_out.shape)

        lstm_out = tf.reshape(fc_out, [shape[0], -1, flags.FLAGS.n_classes],
                              name='lstm_out')  # [batch, width, n_classes]
        logging.info('lstm_out shape: %s' % lstm_out.shape)

        # Swap batch and time axis
        logprob = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

        return logprob


def create_loss(sparse_code_target, logprob, seq_len_inputs):
    with tf.control_dependencies(
            [tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
        loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                  inputs=logprob,
                                  sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                  preprocess_collapse_repeated=False,
                                  ctc_merge_repeated=True,
                                  ignore_longer_outputs_than_inputs=True,
                                  # returns zero gradient in case it happens -> ema loss = NaN
                                  time_major=True)
        loss_ctc = tf.reduce_mean(loss_ctc)
    return loss_ctc


def create_train_op(sparse_code_target, seq_len_inputs, logprob):
    loss_ctc = create_loss(sparse_code_target, logprob, seq_len_inputs)
    tf.losses.add_loss(loss_ctc)

    global_step = tf.train.get_or_create_global_step()
    # Train op
    # --------
    learning_rate = tf.train.exponential_decay(flags.FLAGS.learning_rate, global_step,
                                               flags.FLAGS.learning_decay_steps, flags.FLAGS.learning_decay_rate,
                                               staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_op = slim.learning.create_train_op(total_loss=tf.losses.get_total_loss(), optimizer=optimizer,
                                             update_ops=update_ops)
    tf.summary.scalar('tf-crnn/ctc_loss', loss_ctc)
    return train_op


def create_metrics(logprob, seq_len_inputs, sparse_code_target):
    with tf.name_scope('decode_conversion'):
        sparse_code_pred, log_probability = tf.nn.ctc_greedy_decoder(logprob,
                                                                     sequence_length=tf.cast(
                                                                         seq_len_inputs,
                                                                         tf.int32))
        sparse_code_pred = sparse_code_pred[0]

    with tf.name_scope('evaluation'):
        sparse_code_target = tf.cast(sparse_code_target, dtype=tf.int64)
        edit_distance = tf.edit_distance(sparse_code_pred, sparse_code_target)

        CER = tf.metrics.mean(edit_distance, name='CER')
        sequence_accuracy = tf.metrics.mean(tf.cast(tf.equal(edit_distance, 0), tf.float32))

        eval_metric_ops = {
            'CER': CER,
            'SequenceAccuracy': sequence_accuracy
        }

        return slim.metrics.aggregate_metric_map(eval_metric_ops)
