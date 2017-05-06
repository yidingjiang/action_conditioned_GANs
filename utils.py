import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import gfile
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL

DNA_KERN_SIZE = 5
RELU_SHIFT = 1e-12

def build_all_mask(num_frame):
    masks = []
    for i in range(num_frame):
        m = [0]*num_frame
        m[i]=1
        masks.append(m)
    return np.array(masks).astype(bool)

# def get_batch(sess, img_tensor, action_state_tensor, batch_size, next_state_tensor=None):
#     if next_state_tensor is None:
#         img, action = sess.run([img_tensor, action_state_tensor])
#         return img[:,0,:,:,:], img[:,1,:,:,:], action, None
#     else:
#         img, action, state = sess.run([img_tensor, action_state_tensor, next_state_tensor])
#         return img[:,0,:,:,:], img[:,1,:,:,:], action, state

def get_batch(sess, img_tensor, action_state_tensor, batch_size, next_state_tensor=None):
    img, action, state = sess.run([img_tensor, action_state_tensor, next_state_tensor])
    return img[:,:,:,:,:], img[:,:,:,:,:], action, state


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
        for j in range(int(vid.shape[0])):
            save_path = os.path.join(vid_folder, 'frame{:d}.png'.format(j))
            frame = vid[j]
            plt.imsave(save_path, frame)
        vid = generated_sample[i]
        for j in range(int(vid.shape[0])):
            save_path = os.path.join(vid_folder, 'generated{:d}.png'.format(j))
            frame = vid[j]
            plt.imsave(save_path, frame)
        vid = gt[i]
        for j in range(int(vid.shape[0])):
            save_path = os.path.join(vid_folder, 'ground_truth{:d}.png'.format(j))
            frame = vid[j]
            plt.imsave(save_path, frame)


def build_psnr(true, pred):
    return 10.0 * tf.log(1.0 / tf.losses.mean_squared_error(true, pred)) / tf.log(10.0)


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



def build_generator_transform(images, actions, batch_size, reuse=False, color_channels=3, num_masks=1, ksize=5):
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
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv2',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)

        state_out = slim.conv2d(
            out,
            32,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='sconv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        state_out = slim.conv2d(
            state_out,
            16,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='sconv4',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        print(state_out.shape)
        state_out = slim.conv2d(
            state_out,
            5,
            [4, 4],
            activation_fn=None,
            stride=1,
            scope='sconv5',
            padding='VALID',
            normalizer_fn=None,
            reuse=reuse)

        out = slim.conv2d_transpose(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv3',
            padding='SAME',
            normalizer_fn=slim.batch_norm,
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            ksize*ksize,
            [5, 5],
            activation_fn=None,
            stride=2,
            scope='tconv4',
            padding='SAME',
            reuse=reuse)

        out = tf.nn.softmax(out, dim=-1, name=None)
        input_extracted = tf.extract_image_patches(images,
                                                    ksizes=[1, ksize, ksize, 1],
                                                    strides=[1, 1, 1, 1],
                                                    rates=[1, 1, 1, 1],
                                                    padding='SAME')
        print(input_extracted.get_shape())
        input_extracted = tf.reshape(input_extracted,
                                       [batch_size, 64, 64, ksize*ksize, 3])
        out = tf.stack([out]*3, axis=4)
        out *= input_extracted
        out = tf.reduce_sum(out, 3)

    return out, tf.squeeze(state_out)

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


'''
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
    image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
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
        capacity=100 * batch_size)
    action_state = tf.concat(values=[action_batch, state_batch], axis=2)
    return image_batch, action_state
  else:
    image_batch = tf.train.batch(
        [image_seq],
        batch_size,
        num_threads=batch_size,
        capacity=100 * batch_size)
    zeros_batch = tf.zeros([batch_size, sequence_length, 2 * STATE_DIM])
    return image_batch, zeros_batch

