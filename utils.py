import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import argparse

DNA_KERN_SIZE = 5
RELU_SHIFT = 1e-12

def get_batch(sess, img_tensor, action_state_tensor, batch_size):
    img, action = sess.run([img_tensor, action_state_tensor])
    return img[:,0,:,:,:], img[:,1,:,:,:], action

def save_samples(output_path, input_sample, generated_sample, gt, sample_number):
    input_sample = (255. / 2) * (input_sample + 1.)
    input_sample = input_sample.astype(np.uint8)
    generated_sample = (255. / 2) * (generated_sample + 1.)
    generated_sample = generated_sample.astype(np.uint8)
    gt = (255. / 2) * (gt + 1.)
    gt = gt.astype(np.uint8)
    save_folder =  os.path.join(
        output_path,
        'sample{:d}'.format(sample_number))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i in range(input_sample.shape[0]):
        vid_folder = os.path.join(save_folder, 'vid{:d}'.format(i))
        if not os.path.exists(vid_folder):
            os.makedirs(vid_folder)
        vid = input_sample[i]
        for j in range(int(vid.shape[2] / 3)):
            save_path = os.path.join(vid_folder, 'frame{:d}.png'.format(j))
            frame = vid[:,:,3*j:3*(j+1)]
            plt.imsave(save_path, frame[:,:,::-1])
        vid = generated_sample[i]
        for j in range(int(vid.shape[2] / 3)):
            save_path = os.path.join(vid_folder, 'generated{:d}.png'.format(j))
            frame = vid[:,:,3*j:3*(j+1)]
            plt.imsave(save_path, frame[:,:,::-1])
        vid = gt[i]
        for j in range(int(vid.shape[2] / 3)):
            save_path = os.path.join(vid_folder, 'ground_truth{:d}.png'.format(j))
            frame = vid[:,:,3*j:3*(j+1)]
            plt.imsave(save_path, frame[:,:,::-1])

#====================================================model utils====================================================

def cdna_transformation(prev_image, cdna_input, num_masks, color_channels):
    """
    Copied and modified from:
    https:https://github.com/tensorflow/models/tree/master/video_prediction

    Apply convolutional dynamic neural advection to previous image.

    Args:
        prev_image: previous image to be transformed.
        cdna_input: hidden lyaer to be used for computing CDNA kernels.
        num_masks: the number of masks and hence the number of CDNA transformations.
        color_channels: the number of color channels in the images.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    batch_size = int(cdna_input.get_shape()[0])

      # Predict kernels using linear function of last hidden layer.
    cdna_kerns = slim.layers.fully_connected(
        cdna_input,
        DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
        scope='cdna_params',
        activation_fn=None)

      # Reshape and normalize.
    cdna_kerns = tf.reshape(
        cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
    norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
    cdna_kerns /= norm_factor

    cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
    cdna_kerns = tf.split(axis=0, num_or_size_splits=batch_size, value=cdna_kerns)
    prev_images = tf.split(axis=0, num_or_size_splits=batch_size, value=prev_image)

      # Transform image.
    transformed = []
    for kernel, preimg in zip(cdna_kerns, prev_images):
        kernel = tf.squeeze(kernel)
        if len(kernel.get_shape()) == 3:
            kernel = tf.expand_dims(kernel, -1)
        transformed.append(
            tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
    transformed = tf.concat(axis=0, values=transformed)
    print(transformed.get_shape())
    transformed = tf.split(axis=3, num_or_size_splits=num_masks, value=transformed)
    print(len(transformed))
    return transformed

def build_generator_cdna(images, actions, batch_size, reuse=False, color_channels=3, num_masks=1):
    with tf.variable_scope('g', reuse=reuse):
        out = slim.conv2d(
            images,
            32,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            256,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = tf.concat(values=[out, actions], axis=3)
        out = slim.conv2d_transpose(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            32,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        print(out.get_shape())
        cdna_input = tf.reshape(out, [int(batch_size), -1])
        print(cdna_input.get_shape())
        transformed = cdna_transformation(images, cdna_input, num_masks, color_channels)
    return tf.squeeze(transformed)

def build_generator(images, actions, reuse=False):
    with tf.variable_scope('g', reuse=reuse):
        out = slim.conv2d(
            images,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            256,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            512,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = tf.concat(values=[out, actions], axis=3)
        out = slim.conv2d_transpose(
            out,
            256,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            3,
            [5, 5],
            activation_fn=tf.tanh,
            stride=2,
            scope='tconv4',
            padding='SAME',
            reuse=reuse)
    return out

def decoder_block(input,
                    name,
                    output_fn=tf.tanh,
                    reuse=False):
    out = slim.conv2d_transpose(
        input,
        256,
        [5, 5],
        activation_fn=tf.nn.relu,
        stride=2,
        scope= name + 'tconv1',
        padding='SAME',
        normalizer_fn=slim.batch_norm,
        reuse=reuse)
    out = slim.conv2d_transpose(
        out,
        128,
        [5, 5],
        activation_fn=tf.nn.relu,
        stride=2,
        scope= name + 'tconv2',
        padding='SAME',
        normalizer_fn=slim.batch_norm,
        reuse=reuse)
    out = slim.conv2d_transpose(
        out,
        64,
        [5, 5],
        activation_fn=tf.nn.relu,
        stride=2,
        scope= name + 'tconv3',
        padding='SAME',
        normalizer_fn=slim.batch_norm,
        reuse=reuse)
    out = slim.conv2d_transpose(
        out,
        3,
        [5, 5],
        activation_fn=output_fn,
        stride=2,
        scope= name + 'tconv4',
        padding='SAME',
        reuse=reuse)
    return out

def build_generator_atn(inputs,
                        actions,
                        reuse=False):
    with tf.variable_scope('g', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            256,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            512,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = tf.concat(values=[out, actions], axis=3)

        foreground = decoder_block(out, name='foreground')
        mask = decoder_block(out, name='mask', output_fn=tf.nn.sigmoid)
        background = decoder_block(out, name='background')

        return foreground, mask, background

def build_generator_residual(inputs,
                                actions,
                                reuse=False):
    with tf.variable_scope('g', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            256,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            512,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = tf.concat(values=[out, actions], axis=3)
        foreground = decoder_block(out, name='foreground')
        mask = decoder_block(out, name='mask', output_fn=tf.nn.sigmoid)
        negative_mask = tf.ones_like(mask) - mask
        out = tf.multiply(mask, foreground) + tf.multiply(negative_mask, inputs)
        return out

def build_discriminator(inputs,
                        actions,
                        reuse=False):
    with tf.variable_scope('d', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv1',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv2',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            tf.concat(values=[out, actions], axis=3),
            128,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv3',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            256,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv4',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            512,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv5',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d(
            out,
            1,
            [2, 2],
            activation_fn=None,
            stride=1,
            scope='conv6',
            reuse=reuse)
    return out


def build_gdl(g_out, next_frames, alpha):
    '''
    Copied from:
    https://github.com/dyelax/Adversarial_Video_Generation/
    '''
    pos = tf.constant(np.identity(3), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
    strides = [1, 1, 1, 1]
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(g_out, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(g_out, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(next_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(next_frames, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    return tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha))

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#====================================================Loss====================================================

def wasserstein_discriminator_loss(d_real, d_sample):
    return (tf.reduce_mean(d_real) - tf.reduce_mean(d_sample))

def wasserstein_generator_loss(d_sample):
    return  tf.reduce_mean(d_sample)

def gan_discriminator_loss(d_real_logit, d_sample_logit, real_prob=0.8):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit, labels=real_prob*tf.ones_like(d_real_logit)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_sample_logit, labels=(1-real_prob)*tf.ones_like(d_sample_logit)))
    return d_loss_real + d_loss_sample

def gan_generator_loss(d_sample_logit):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_sample_logit, labels=tf.ones_like(d_sample_logit)))

#=====================================================layers=================================================

def conv2d(incoming_layer, filter_shape,
        strides, name, group, act=tf.identity,
        padding='SAME'):
    with tf.variable_scope(name) as vs:
        W = tf.get_variable(name='W', shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=(filter_shape[-1]),
            initializer=tf.constant_initializer(value=0.0))
    return act(tf.nn.conv2d(incoming_layer, W, strides=strides, padding=padding) + b)


def deconv3d(incoming_layer, filter_shape, output_shape,
        strides, name,  group, act=tf.identity,
        padding='SAME'):
    with tf.variable_scope(name) as vs:
        W = tf.get_variable(name='W', shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=(filter_shape[-2]),
            initializer=tf.constant_initializer(value=0.0))
    rtn = tf.nn.conv3d_transpose(incoming_layer, W, output_shape=output_shape, strides=strides, padding=padding)
    rtn.set_shape([None] + output_shape[1:])
    return act(tf.nn.bias_add(rtn, b))


def conv3d(incoming_layer, filter_shape,
        strides, name, group, act=tf.identity,
        padding='SAME'):
    with tf.variable_scope(name) as vs:
        W = tf.get_variable(name='W', shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b', shape=(filter_shape[-1]),
            initializer=tf.constant_initializer(value=0.0))
    return act(tf.nn.conv3d(incoming_layer, W, strides=strides, padding=padding, name=None) + b)


def batchnorm(incoming_layer, phase, name, group):
    dummy = dummyLayer(incoming_layer)
    network = BatchNormLayer(layer = dummy,
                            is_train = phase,
                            name = name)
    return network.outputs


class dummyLayer:
    def __init__(self, output):
        self.outputs = output
        self.all_layers = []
        self.all_params = []
        self.all_drop = []


class BatchNormLayer():
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.constant_initializer(value=0.0),
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
        name ='g_bn_layer',
    ):
        self.inputs = layer.outputs
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )

            ## 2.
            moving_mean_init = tf.constant_initializer(0.0)
            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=moving_mean_init,
                                      trainable=False,)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False,)#   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            else:
                self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )

            variables = [beta, gamma, moving_mean, moving_variance]

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )