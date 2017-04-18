import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import argparse

def get_batch(vid_data, action_data, batch_size, start, end, history_length):
    vid_indices = sorted(random.sample(range(start, end), batch_size))
    start_idx = np.random.choice(3)
    batch = vid_data[vid_indices][:,:,:,3*start_idx:3*(start_idx + history_length + 1)]
    batch = batch / 255. # normalize and center
    action_batch = np.squeeze(action_data[vid_indices])[:,5*start_idx]
    input_batch = batch[:,:,:,:3*history_length]
    next_frame_batch = batch[:,:,:,3*history_length:]
    return input_batch, next_frame_batch, action_batch

def save_samples(output_path, input_sample, generated_sample, gt, sample_number):
    input_sample = 255. * input_sample
    input_sample = input_sample.astype(np.uint8)
    generated_sample = 255. * generated_sample
    generated_sample = generated_sample.astype(np.uint8)
    gt = 255. * gt
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
        num_frames = int(vid.shape[2] / 3)
        for j in range(num_frames):
            save_path_input = os.path.join(vid_folder, 'frame{:d}.png'.format(j))
            frame = vid[:,:,3*j:3*(j+1)]
            plt.imsave(save_path_input, frame[:,:,::-1])
        save_path_generated = os.path.join(vid_folder, 'generated0.png')
        plt.imsave(save_path_generated, generated_sample[i][:,:,::-1])
        save_path_gt = os.path.join(vid_folder, 'gt0.png')
        plt.imsave(save_path_gt, gt[i][:,:,::-1])

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
#todo: add Xavier initialization
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
    #a class designed to fool tensorlayer...
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
        # print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" %
        #                     (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            # if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
            #     beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)#, restore=restore)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )#restore=restore)

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