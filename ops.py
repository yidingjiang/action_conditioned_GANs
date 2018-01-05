import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import gfile
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import imageio

DNA_KERN_SIZE = 5
RELU_SHIFT = 1e-12

def get_batch(sess, img_tensor, action_state_tensor, batch_size, next_state_tensor=None):
    img, action, state = sess.run([img_tensor, action_state_tensor, next_state_tensor])
    return img[:,:,:,:,:], img[:,:,:,:,:], action, state

def build_psnr(true, pred):
    return 10.0 * tf.log(1.0 / tf.losses.mean_squared_error(true, pred)) / tf.log(10.0)

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def build_g_adv_loss(d_out_gen, arg_loss):
    if arg_loss == 'bce':
        return tf.losses.sigmoid_cross_entropy(
            tf.ones_like(d_out_gen), d_out_gen)
    elif arg_loss == 'wass':
        return tf.reduce_mean(d_out_gen)
    else:
    	raise ValueError('unexpected loss argument')

def build_d_loss(d_out_direct, d_out_gen, arg_loss):
    if arg_loss == 'bce':
        d_direct_loss = tf.losses.sigmoid_cross_entropy(
            0.9 * tf.ones_like(d_out_direct), d_out_direct)
        d_gen_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(d_out_gen), d_out_gen)
    elif arg_loss == 'wass':
        d_direct_loss = tf.reduce_mean(d_out_direct)
        d_gen_loss = -tf.reduce_mean(d_out_gen)
    else:
    	raise ValueError('unexpected loss argument')
    d_direct_loss_summary = tf.summary.scalar('discriminator_direct_loss', d_direct_loss)
    d_gen_loss_summary = tf.summary.scalar('discriminator_gen_loss', d_gen_loss)
    return d_direct_loss + d_gen_loss

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
    transformed = tf.split(axis=3, num_or_size_splits=num_masks, value=transformed)

    return transformed

def build_gdl(g_out, next_frames, alpha=1):
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

'''
Data loading functions
Adapted from
https://github.com/tensorflow/models/blob/master/video_prediction/prediction_input.py
'''

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Dimension of the state and action.
STATE_DIM = 5


def build_tfrecord_input(batch_size,
                         data_dir,
                         sequence_length,
                         train_val_split,
                         use_state,
                         training=True):
  """Create input tfrecord tensors.

  Args:
    training: training or validation data.
  Returns:
    list of tensors corresponding to images, actions, and states. The images
    tensor is 5D, batch x time x height x width x channels. The state and
    action tensors are 3D, batch x time x dimension.
  Raises:
    RuntimeError: if no files found.
  """
  filenames = gfile.Glob(os.path.join(data_dir, '*'))
  if not filenames:
    raise RuntimeError('No data files found.')
  index = int(np.floor(train_val_split * len(filenames)))
  if training:
    filenames = filenames[:index]
  else:
    filenames = filenames[index:]
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  image_seq, state_seq, action_seq = [], [], []

  for i in range(6, 20, 2):
    image_name = 'move/' + str(i) + '/image/encoded'
    action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
    state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'
    if use_state:
      features = {image_name: tf.FixedLenFeature([1], tf.string),
                  action_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
                  state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)}
    else:
      features = {image_name: tf.FixedLenFeature([1], tf.string)}
    features = tf.parse_single_example(serialized_example, features=features)

    image_buffer = tf.reshape(features[image_name], shape=[])
    image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
    image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

    if IMG_HEIGHT != IMG_WIDTH:
      raise ValueError('Unequal height and width unsupported')

    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
    image = tf.image.resize_area(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (tf.cast(image, tf.float32) / (255.0 / 2.)) - 1.
    image_seq.append(image)

    if use_state:
      state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
      state_seq.append(state)
      action = tf.reshape(features[action_name], shape=[1, STATE_DIM])
      action_seq.append(action)

  image_seq = tf.concat(axis=0, values=image_seq)
  if use_state:
    state_seq = tf.concat(axis=0, values=state_seq)
    action_seq = tf.concat(axis=0, values=action_seq)

    [image_batch, action_batch, state_batch] = tf.train.batch(
        [image_seq, action_seq, state_seq],
        batch_size,
        num_threads=batch_size,
        capacity=500 * batch_size)
    action_state = tf.concat(values=[action_batch, state_batch], axis=2)
    return image_batch, action_state
  else:
    image_batch = tf.train.batch(
        [image_seq],
        batch_size,
        num_threads=batch_size,
        capacity=500 * batch_size)
    zeros_batch = tf.zeros([batch_size, sequence_length, 2 * STATE_DIM])
    return image_batch, zeros_batch