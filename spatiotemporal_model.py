import argparse
import tensorflow as tf
import numpy as np
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.convolutional import Deconvolution3D
from utils import lrelu, build_tfrecord_input


class Model():
    def __init__(self, sess, sequence):
        self.sess = sess
        '''
        sequence = tf.placeholder(
            tf.float32,
            [None, 20, 64, 64, 3])
        '''
        input_images = sequence[:,:4,:,:,:]

        with tf.variable_scope('g'):
            self.g_out = self._build_generator(input_images)
        with tf.variable_scope('d'):
            d_conv_layers = self._build_d_conv_layers()
            self.d_real = self._build_discriminator(sequence, d_conv_layers)
        with tf.variable_scope('d', reuse=True):
            self.d_gen = self._build_discriminator(self.g_out, d_conv_layers)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        # Batch norm update ops
        self.g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'g')
        self.d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'd')

        # Define loss for both d and g
        self.g_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(self.d_gen), self.d_gen)
        d_real_loss = tf.losses.sigmoid_cross_entropy(
            0.9 * tf.ones_like(self.d_real), self.d_real)
        d_gen_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(self.d_gen), self.d_gen)
        self.d_loss = d_real_loss + d_gen_loss

        # Make batch norm updates dependencies of optimization ops
        with tf.control_dependencies(self.g_update_ops):
            self.g_opt_op = tf.train.AdamOptimizer(1e-3, name='g_opt').minimize(
                self.g_loss, var_list=self.g_vars)
        with tf.control_dependencies(self.d_update_ops):
            self.d_opt_op = tf.train.AdamOptimizer(1e-3, name='d_opt').minimize(
                self.d_loss, var_list=self.d_vars)


    def train_g(self):
        _, g_loss = self.sess.run([self.g_opt_op, self.g_loss])
        return g_loss


    def train_d(self):
        _, d_loss = self.sess.run([self.d_opt_op, self.d_loss])
        return d_loss


    def _build_d_conv_layers(self):
        d_conv_specs=[
            (32, [1, 3, 3], [1, 2, 2]),
            (64, [2, 3, 3], [2, 2, 2]),
            (128, [2, 3, 3], [2, 2, 2]),
            (256, [2, 3, 3], [2, 2, 2]),
            (1, [2, 3, 3], [1, 1, 1])
        ]
        d_conv_layers = []
        for i, spec in enumerate(d_conv_specs):
            layer = Conv3D(
                spec[0],
                spec[1],
                strides=spec[2],
                activation=lrelu if i < len(d_conv_specs) - 1 else None,
                name='conv{:d}'.format(i))
            d_conv_layers.append(layer)
        return d_conv_layers


    def _build_discriminator(self, output, d_conv_layers):
        out = output
        for i, layer in enumerate(d_conv_layers):
            out = layer(out)
            if i < len(d_conv_layers) - 1:
                out = tf.layers.batch_normalization(
                    out,
                    training=True,
                    name='bn{:d}'.format(i))
        out = tf.squeeze(out)
        return out


    def _build_generator(self, img_input):
        batch_size = tf.shape(img_input)[0]
        conv_specs = [
            (32, [1, 3, 3]),
            (64, [1, 3, 3]),
            (128, [1, 3, 3]),
            (256, [1, 3, 3]),
            (32, [4, 1, 1])
        ]
        out = img_input
        for i, spec in enumerate(conv_specs):
            out = Conv3D(
                spec[0],
                spec[1],
                activation='relu',
                padding='same' if i < len(conv_specs) - 1 else 'valid',
                name='conv{:d}'.format(i))(out)
            out = tf.layers.batch_normalization(
                out,
                training=True,
                name='bn{:d}'.format(i))
        out = Deconvolution3D(
            32,
            [4, 1, 1],
            activation='relu',
            output_shape=[None, 4, 64, 64, 32],
            name='tconv1')(out)
        out = tf.layers.batch_normalization(
            out,
            training=True,
            name='bn5')
        out = Deconvolution3D(
            32,
            [4, 1, 1],
            strides=[2, 1, 1],
            activation='relu',
            padding='same',
            output_shape=[None, 8, 64, 64, 32],
            name='tconv2')(out)
        out = tf.layers.batch_normalization(
            out,
            training=True,
            name='bn6')
        out = Deconvolution3D(
            25,
            [4, 1, 1],
            strides=[2, 1, 1],
            activation='relu',
            padding='same',
            output_shape=[None, 16, 64, 64, 25],
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
        for i in range(16):
            normalized_transform = tf.nn.l2_normalize(out[:,i,:,:,:], dim=3)
            repeated_transform = tf.stack(3 * [normalized_transform], axis=4)
            frame_i = tf.reduce_sum(repeated_transform * patches, axis=3)
            output_frames.append(frame_i)

        combined = [img_input, tf.stack(output_frames, axis=1)]
        final_output = tf.concat(values=combined, axis=1)
        # batchnorm update ops
        return final_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()
    sequence, actions = build_tfrecord_input(4, args.input_path, 20, 1, True)
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        m = Model(sess, sequence)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('Model construction completed.')
        for i in range(args.iterations):
            g_loss = m.train_g()
            d_loss = m.train_d()
            print('Iteration {:d}: {:2f} {:2f}'.format(i, g_loss, d_loss))
