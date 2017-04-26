import tensorflow as tf
import numpy as np
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.convolutional import Deconvolution3D


def build_generator(img_input):
    #img_input = tf.placeholder(tf.float32, [None, 2, 64, 64, 3])
    batch_size = tf.shape(img_input)[0]
    conv_specs = [
        (32, [1, 3, 3]),
        (64, [1, 3, 3]),
        (128, [1, 3, 3]),
        (256, [1, 3, 3]),
        (32, [2, 1, 1])
    ]
    with tf.variable_scope('g'):
        out = img_input
        bn_updates = []
        for i, spec in conv_specs:
            out = Conv3D(
                spec[0],
                spec[1],
                activation='relu',
                padding='same',
                name='conv{:d}'.format(i))(out)
            out = tf.layers.batch_normalization(out, training=True)
        out = Deconvolution3D(
            32,
            [2, 1, 1],
            activation='relu',
            output_shape=[None, 2, 64, 64, 32],
            name='tconv1')(out)
        out = tf.layers.batch_normalization(out, training=True)
        out = Deconvolution3D(
            32,
            [2, 1, 1],
            strides=[2, 1, 1],
            activation='relu',
            padding='same',
            output_shape=[None, 4, 64, 64, 32],
            name='tconv2')(out)
        out = tf.layers.batch_normalization(out, training=True)
        out = Deconvolution3D(
            25,
            [2, 1, 1],
            strides=[2, 1, 1],
            activation='relu',
            padding='same',
            output_shape=[None, 8, 64, 64, 25],
            name='tconv3')(out)

        prev_frame = img_input[:,1,:,:,:]
        patches = tf.extract_image_patches(
            prev_frame,
            ksizes=[1, 5, 5, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches = tf.reshape(patches, [batch_size, 64, 64, 25, 3])

        output_frames = []
        for i in range(8):
            normalized_transform = tf.nn.l2_normalize(out[:,i,:,:,:], dim=3)
            repeated_transform = tf.stack(3 * [normalized_transform], axis=4)
            frame_i = tf.reduce_sum(repeated_transform * patches, axis=3)
            output_frames.append(frame_i)

        final_output = tf.stack(output_frames, axis=1)
        # batchnorm update ops
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return final_output
