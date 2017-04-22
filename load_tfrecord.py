'''
Adapted from
https://github.com/tensorflow/models/blob/master/video_prediction/prediction_input.py
'''

import os

import numpy as np
import tensorflow as tf

from tensorflow import gfile

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

  for i in range(sequence_length):
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

