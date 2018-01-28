import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

from ops import lrelu

def build_generator(images, actions, reuse=False):
    with tf.variable_scope('g', reuse=reuse):
        with slim.argscope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, stride=2, 
                           padding='SAME', normalizer_fn=slim.batch_norm, reuse=reuse):
            out = slim.conv2d(images, 64, [5, 5], scope='conv1')
            out = slim.conv2d(out, 128, [5, 5], scope='conv2')
            out = slim.conv2d(out, 256, [5, 5], scope='conv3')
            out = slim.conv2d(out, 512, [5, 5], scope='conv4')
            out = tf.concat(values=[out, actions], axis=3)
            out = slim.conv2d_transpose(out, 256, [5, 5], scope='tconv1')
            out = slim.conv2d_transpose(out, 128, [5, 5], scope='tconv2')
            out = slim.conv2d_transpose(out, 64, [5, 5], scope='tconv3')
            out = slim.conv2d_transpose(out, 3, [5, 5], activation_fn=tf.tanh, 
                                        scope='tconv4', normalizer_fn=None)
            return out

def build_generator_transform(images, 
                              actions, 
                              batch_size, 
                              reuse=False, 
                              color_channels=3, 
                              ksize=5):
    with tf.variable_scope('g', reuse=reuse):
        with slim.argscope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, stride=2, 
                           padding='SAME', normalizer_fn=slim.batch_norm, reuse=reuse):

            out = slim.conv2d(images, 32, [5, 5], scope='conv1')
            out = slim.conv2d(out, 64, [5, 5], scope='conv2')
            out = slim.conv2d(out, 128, [5, 5], scope='conv3')
            out = slim.conv2d(out, 256, [5, 5], scope='conv4')
            out = tf.concat(values=[out, actions], axis=3)
            out = slim.conv2d_transpose(out, 128, [5, 5], scope='tconv1')
            out = slim.conv2d_transpose(out, 128, [5, 5], scope='tconv2')

            state_out = slim.conv2d(out, 32, [3, 3], scope='sconv3')
            state_out = slim.conv2d(state_out, 16, [3, 3], scope='sconv4')
            state_out = slim.conv2d(state_out,
                                    5,
                                    [4, 4],
                                    activation_fn=None,
                                    stride=1,
                                    scope='sconv5',
                                    padding='VALID',
                                    normalizer_fn=None)

            out = slim.conv2d_transpose(out, 128, [5, 5], scope='tconv3')
            out = slim.conv2d_transpose(out,
                                        ksize*ksize,
                                        [5, 5],
                                        activation_fn=None,
                                        scope='tconv4',
                                        normalizer_fn=None)
            out = tf.nn.softmax(out, dim=-1, name=None)

            input_extracted = tf.extract_image_patches(images,
                                                       ksizes=[1, ksize, ksize, 1],
                                                       strides=[1, 1, 1, 1],
                                                       rates=[1, 1, 1, 1],
                                                       padding='SAME')
            input_extracted = tf.reshape(input_extracted,
                                         [batch_size, 64, 64, ksize*ksize, 3])

            out = tf.stack([out]*3, axis=4)
            out *= input_extracted
            out = tf.reduce_sum(out, 3)

            return out, tf.squeeze(state_out)

def build_discriminator(inputs,
                        actions,
                        reuse=False):
    with tf.variable_scope('d', reuse=reuse):
        with slim.argscope([slim.conv2d], activation_fn=lrelu, stride=2, 
                           padding='SAME', normalizer_fn=slim.batch_norm, reuse=reuse):
            out = slim.conv2d(inputs, 64, [5, 5], scope='conv1')
            out = slim.conv2d(out, 128, [5, 5], scope='conv2')
            out = slim.conv2d(tf.concat(values=[out, actions], axis=3), 128, [5, 5], scope='conv3')
            out = slim.conv2d(out, 256, [5, 5], scope='conv4')
            out = slim.conv2d(out, 512, [5, 5], scope='conv5')
            out = slim.conv2d(out, 1, [2, 2], activation_fn=None, 
                              stride=1, scope='conv6', reuse=reuse)
            return out