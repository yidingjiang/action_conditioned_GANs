import argparse
import os
import tensorflow as tf
import numpy as np
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.convolutional import Deconvolution3D
from utils import lrelu, build_tfrecord_input, save_samples


class Model():
    def __init__(self, sess, sequence, actions, arg_g_loss, arg_actions):
        self.sess = sess
        self.sequence = sequence
        input_images = self.sequence[:,:2,:,:,:]
        tiled_actions = None
        d_actions = None
        if arg_actions:
            actions_by_ts = []
            for i in range(1, 9):
                actions_by_ts.append(actions[:,i,:])
            reshaped_actions = tf.concat(
                axis=1, values=actions_by_ts)[:,None,None, None,:]
            tiled_actions = tf.tile(
                reshaped_actions, [1, 1, 64, 64, 1])
            d_actions = tf.tile(actions[:,:,None,None,:], [1, 1, 64, 64, 1])

        with tf.variable_scope('g'):
            self.g_out = self._build_generator(input_images, tiled_actions)
        with tf.variable_scope('d'):
            d_conv_layers = self._build_d_conv_layers()
            self.d_real = self._build_discriminator(self.sequence, d_conv_layers, d_actions)
        with tf.variable_scope('d', reuse=True):
            self.d_gen = self._build_discriminator(self.g_out, d_conv_layers, d_actions)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        # self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]
        # Batch norm update ops
        self.g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'g')
        self.d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'd')

        # Define loss for both d and g
        l_ord = 2
        g_lp_loss = tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.g_out - self.sequence)**l_ord, axis=[1, 2, 3, 4]))

        g_adv_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(self.d_gen), self.d_gen)
        if arg_g_loss == 'l2':
            self.g_loss = g_lp_loss
        elif arg_g_loss == 'adv':
            self.g_loss = g_adv_loss
        else:
            self.g_loss = .005 * g_lp_loss + g_adv_loss
        d_real_loss = tf.losses.sigmoid_cross_entropy(
            0.9 * tf.ones_like(self.d_real), self.d_real)
        d_gen_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(self.d_gen), self.d_gen)
        self.d_loss = d_real_loss + d_gen_loss
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('g_lp_loss', g_lp_loss)
        tf.summary.scalar('g_adv_loss', g_adv_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_gen_loss', d_gen_loss)
        tf.summary.scalar('d_loss', self.d_loss)

        # Make batch norm updates dependencies of optimization ops
        with tf.control_dependencies(self.g_update_ops):
            self.g_opt_op = tf.train.AdamOptimizer(1e-3, name='g_opt').minimize(
                self.g_loss, var_list=self.g_vars)
        with tf.control_dependencies(self.d_update_ops):
            self.d_opt_op = tf.train.AdamOptimizer(1e-3, name='d_opt').minimize(
                self.d_loss, var_list=self.d_vars)

        self.merged_summaries = tf.summary.merge_all()


    def train_g(self, output=False):
        if output:
            _, g_loss, seq, g_out = self.sess.run([self.g_opt_op, self.g_loss, self.sequence, self.g_out])
            return g_loss, seq, g_out
        else:
            _, g_loss = self.sess.run([self.g_opt_op, self.g_loss])
            return g_loss, None, None


    def train_d(self, summarize=False):
        if summarize:
            _, d_loss, summ = self.sess.run([self.d_opt_op, self.d_loss, self.merged_summaries])
            return d_loss, summ
        else:
            _, d_loss = self.sess.run([self.d_opt_op, self.d_loss])
            return d_loss, None


    def _build_d_conv_layers(self):
        d_conv_specs=[
            (32, [3, 3, 3], [2, 2, 2]),
            (64, [3, 3, 3], [2, 2, 2]),
            (128, [3, 3, 3], [2, 2, 2]),
            (256, [3, 3, 3], [2, 2, 2]),
            (1, [1, 4, 4], [1, 1, 1])
        ]
        d_conv_layers = []
        for i, spec in enumerate(d_conv_specs):
            layer = Conv3D(
                spec[0],
                spec[1],
                strides=spec[2],
                activation=lrelu if i < len(d_conv_specs) - 1 else None,
                padding='same' if i < len(d_conv_specs) -1 else 'valid',
                name='conv{:d}'.format(i))
            d_conv_layers.append(layer)
        return d_conv_layers


    def _build_discriminator(self, output, d_conv_layers, d_actions):
        out = output
        if d_actions is not None:
            out = tf.concat(values=[out, d_actions], axis=-1)
        for i, layer in enumerate(d_conv_layers):
            out = layer(out)
            if i < len(d_conv_layers) - 1:
                out = tf.layers.batch_normalization(
                    out,
                    training=True,
                    name='bn{:d}'.format(i))
        out = tf.squeeze(out)
        return out


    def _build_generator(self, img_input, tiled_actions):
        ksize = 10
        batch_size = tf.shape(img_input)[0]
        conv_specs = [
            (32, [1, 3, 3]),
            (64, [1, 3, 3]),
            (128, [1, 3, 3]),
            (256, [1, 3, 3]),
            (32, [2, 1, 1])
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
        if tiled_actions is not None:
            out = tf.concat(values=[out, tiled_actions], axis=-1)
        out = Deconvolution3D(
            32,
            [2, 1, 1],
            activation='relu',
            output_shape=[None, 2, 64, 64, 32],
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
            output_shape=[None, 4, 64, 64, 32],
            name='tconv2')(out)
        out = tf.layers.batch_normalization(
            out,
            training=True,
            name='bn6')
        out = Deconvolution3D(
            ksize * ksize,
            [4, 1, 1],
            strides=[2, 1, 1],
            activation=None,
            padding='same',
            output_shape=[None, 8, 64, 64, ksize * ksize],
            name='tconv3')(out)

        prev_frame = img_input[:,1,:,:,:]
        patches = tf.extract_image_patches(
            prev_frame,
            ksizes=[1, ksize, ksize, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        patches = tf.reshape(patches, [batch_size, 64, 64, ksize * ksize, 3])

        output_frames = []
        for i in range(8):
            #normalized_transform = tf.nn.l2_normalize(out[:,i,:,:,:], dim=3)
            normalized_transform = tf.nn.softmax(out[:,i,:,:,:], dim=-1)
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
    parser.add_argument('output_path', type=str)
    parser.add_argument('--g_loss', type=str, default='l2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--actions', action='store_true')
    args = parser.parse_args()
    train_output_path = os.path.join(args.output_path, 'train_output')
    model_dir = os.path.join(args.output_path, 'models')
    log_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    sequence, actions = build_tfrecord_input(
        args.batch_size,
        args.input_path,
        10, .95, True)
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        m = Model(sess, sequence, actions, args.g_loss, args.actions)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        saver = tf.train.Saver()
        print('Model construction completed.')
        for i in range(args.iterations):
            g_loss, seq, g_out = m.train_g(output=(i%100==0))
            d_loss, summ = m.train_d(summarize=(i%100==0))
            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
                save_samples(train_output_path, seq, g_out, i)
                train_writer.add_summary(summ, i)
                train_writer.flush()
            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, 'model{:d}'.format(i)))
