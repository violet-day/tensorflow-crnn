# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from util import seed
from captcha.image import ImageCaptcha
from tensorflow.python.platform import flags

image = ImageCaptcha(height=flags.FLAGS.image_height, width=flags.FLAGS.image_width)

charsets = '0123456789abcdefghijklmnopqrstuvwxyz'


def gen_random_text(charsets, min_len, max_len):
    length = seed.random_integers(low=min_len, high=max_len)
    idxs = seed.randint(low=0, high=len(charsets), size=length)
    str = ''.join([charsets[i] for i in idxs])
    return idxs, str


def gen_img(text, image_shape):
    data = image.generate_image(text)
    data = np.reshape(np.frombuffer(data.tobytes(), dtype=np.uint8), image_shape)
    return data


def batch_gen(batch_size, charsets, min_len, max_len, image_shape, blank_symbol):
    def gen():
        while True:
            batch_labels = []
            batch_images = []
            batch_image_widths = []

            for _ in range(batch_size):
                idxs, text = gen_random_text(charsets, min_len, max_len)
                image = gen_img(text, image_shape)

                pad_size = max_len - len(idxs)
                if pad_size > 0:
                    idxs = np.pad(idxs, pad_width=(0, pad_size), mode='constant', constant_values=blank_symbol)
                batch_image_widths.append(image.shape[1])
                batch_labels.append(idxs)
                batch_images.append(image)

            batch_labels = np.array(batch_labels, dtype=np.int32)
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_image_widths = np.array(batch_image_widths, dtype=np.int32)
            yield batch_images, batch_image_widths, batch_labels

    return gen


def dense_to_sparse(dense_tensor, blank_symbol):
    indices = tf.where(tf.not_equal(dense_tensor, blank_symbol))
    values = tf.gather_nd(dense_tensor, indices)
    sparse_target = tf.SparseTensor(indices, values, dense_tensor.get_shape())
    return sparse_target


def get_input():
    gen = batch_gen(batch_size=flags.FLAGS.batch_size,
                    charsets=charsets,
                    min_len=flags.FLAGS.min_len,
                    max_len=flags.FLAGS.max_len,
                    image_shape=(flags.FLAGS.image_height, flags.FLAGS.image_width,
                                 flags.FLAGS.image_channel),
                    blank_symbol=flags.FLAGS.n_classes)

    images, image_widths, labels = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32, tf.int32)) \
        .make_one_shot_iterator() \
        .get_next()

    images = tf.reshape(images, [flags.FLAGS.batch_size, flags.FLAGS.image_height, flags.FLAGS.image_width,
                                 flags.FLAGS.image_channel])

    image_widths = tf.reshape(image_widths, [flags.FLAGS.batch_size])
    labels = tf.reshape(labels, [flags.FLAGS.batch_size, flags.FLAGS.max_len])
    labels = dense_to_sparse(labels, blank_symbol=flags.FLAGS.n_classes)
    return images, image_widths, labels
