import argparse
import os
import tensorflow as tf
import numpy as np
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.convolutional import Deconvolution3D
from utils import lrelu, build_tfrecord_input, save_samples, build_psnr, build_gdl_3d


class Model():
    def __init__(self, sess, arg_g_loss, arg_gdl, arg_actions, arg_l_ord, arg_dilation, arg_softmax):
        self.sess = sess
        self.sequence = tf.placeholder(tf.float32, [None, 6, 64, 64, 3], name='sequence_ph')
        self.actions = tf.placeholder(tf.float32, [None, 6, 10], name='actions_ph')
        input_images = self.sequence[:,:2,:,:,:]
        tiled_actions = None
        d_actions = None
        if arg_actions:
            actions_by_ts = []
            for i in range(1, 5):
                actions_by_ts.append(self.actions[:,i,:])
            reshaped_actions = tf.concat(
                axis=1, values=actions_by_ts)[:,None,None, None,:]
            tiled_actions = tf.tile(
                reshaped_actions, [1, 1, 64, 64, 1])
            d_actions = tf.tile(self.actions[:,:,None,None,:], [1, 1, 64, 64, 1])

        with tf.variable_scope('g'):
            self.g_out = self._build_generator(input_images, tiled_actions)
        with tf.variable_scope('d'):
            d_conv_layers = self._build_d_conv_layers()
            d_conv_layer_weights = [v for layer in d_conv_layers for v in layer.trainable_weights]
            self.d_real = self._build_discriminator(self.sequence, d_conv_layers, d_actions)
        with tf.variable_scope('d', reuse=True):
            self.d_gen = self._build_discriminator(self.g_out, d_conv_layers, d_actions)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_conv_layer_weights]
        self.g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'g')
        self.d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'd')

        # Define loss for both d and g
        pred_next_frames = self.g_out[:,2:,:,:,:]
        gt_next_frames = self.sequence[:,2:,:,:,:]
        g_lp_loss = tf.reduce_mean(tf.reduce_sum(
            tf.abs(pred_next_frames - gt_next_frames)**arg_l_ord, axis=[1, 2, 3, 4]))

        g_adv_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(self.d_gen), self.d_gen)
        self.gdl = build_gdl_3d(pred_next_frames, gt_next_frames, arg_l_ord)
        if arg_g_loss == 'lp':
            self.g_loss = g_lp_loss
        elif arg_g_loss == 'adv':
            self.g_loss = g_adv_loss
        else:
            self.g_loss = .01 * g_lp_loss + g_adv_loss
        if arg_gdl:
            self.g_loss += .001 * self.gdl

        d_real_loss = tf.losses.sigmoid_cross_entropy(
            0.9 * tf.ones_like(self.d_real), self.d_real)
        d_gen_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(self.d_gen), self.d_gen)
        self.d_loss = d_real_loss + d_gen_loss

        psnr = build_psnr(gt_next_frames, pred_next_frames)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('g_l{:d}_loss'.format(arg_l_ord), g_lp_loss)
        tf.summary.scalar('g_adv_loss', g_adv_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_gen_loss', d_gen_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('psnr', psnr)
        tf.summary.scalar('gdl', self.gdl)

        # Make batch norm updates dependencies of optimization ops
        with tf.control_dependencies(self.g_update_ops):
            self.g_opt_op = tf.train.AdamOptimizer(1e-3, name='g_opt').minimize(
                self.g_loss, var_list=self.g_vars)
        with tf.control_dependencies(self.d_update_ops + self.clip_d):
            self.d_opt_op = tf.train.AdamOptimizer(1e-3, name='d_opt').minimize(
                self.d_loss, var_list=self.d_vars)

        self.merged_summaries = tf.summary.merge_all()


    def train_g(self, sequence, actions, output=False):
        feed_dict = {
            self.sequence: sequence,
            self.actions: actions
        }
        if output:
            _, g_loss, g_out = self.sess.run(
                [self.g_opt_op, self.g_loss, self.g_out],
                feed_dict=feed_dict)
            return g_loss, g_out
        else:
            _, g_loss = self.sess.run(
                [self.g_opt_op, self.g_loss],
                feed_dict=feed_dict)
            return g_loss, None


    def train_d(self, sequence, actions, summarize=False):
        feed_dict = {
            self.sequence: sequence,
            self.actions: actions
        }
        if summarize:
            _, d_loss, summ = self.sess.run(
                [self.d_opt_op, self.d_loss, self.merged_summaries],
                feed_dict=feed_dict)
            return d_loss, summ
        else:
            _, d_loss = self.sess.run(
                [self.d_opt_op, self.d_loss],
                feed_dict=feed_dict)
            return d_loss, None


    def test_batch(self, sequence, actions):
        feed_dict = {
            self.sequence: sequence,
            self.actions: actions
        }
        g_out, summ = self.sess.run([self.g_out, self.merged_summaries], feed_dict=feed_dict)
        return g_out, summ


    def _build_d_conv_layers(self):
        d_conv_specs=[
            (32, [1, 3, 3], [1, 2, 2]),
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
                padding='same' if i < len(d_conv_specs) - 1 else 'valid',
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


    def _build_generator(self, img_input, tiled_actions, arg_dilation, arg_softmax):
        ksize = 10
        batch_size = tf.shape(img_input)[0]
        conv_specs = [
            (32, [1, 3, 3], [1, 1, 1]),
            (64, [1, 3, 3], [1, 2, 2]),
            (128, [1, 3, 3], [1, 4, 4]),
            (256, [1, 3, 3], [1, 6, 6]),
            (32, [2, 1, 1], [1, 1, 1])
        ]
        out = img_input
        for i, spec in enumerate(conv_specs):
            out = Conv3D(
                spec[0],
                spec[1],
                activation='relu',
                padding='same' if i < len(conv_specs) - 1 else 'valid',
                dilation_rate=spec[2] if arg_dilation else None,
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
            ksize * ksize,
            [4, 1, 1],
            strides=[2, 1, 1],
            activation=None,
            padding='same',
            output_shape=[None, 4, 64, 64, ksize * ksize],
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
        for i in range(4):
            if arg_softmax:
                normalized_transform = tf.nn.softmax(out[:,i,:,:,:], dim=-1)
            else:
                pos_out = tf.nn.relu(out[:,i,:,:,:] - 1e-12) + 1e-12
                normalizer = tf.reduce_sum(pos_out, -1, keep_dims=True)
                normalized_transform = pos_out / normalizer
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
    parser.add_argument('--g_loss', type=str, default='lp')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--actions', action='store_true')
    parser.add_argument('--gdl', action='store_true')
    parser.add_argument('--l_ord', type=int, default=2)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--softmax', action='store_true')
    args = parser.parse_args()
    train_output_path = os.path.join(args.output_path, 'train_output')
    test_output_path = os.path.join(args.output_path, 'test_output')
    model_dir = os.path.join(args.output_path, 'models')
    log_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    sequence, actions = build_tfrecord_input(
        args.batch_size,
        args.input_path,
        20, .95, True)
    test_sequence, test_actions = build_tfrecord_input(
        args.batch_size,
        args.input_path,
        20, .95, True, training=False)
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        m = Model(sess, args.g_loss, args.gdl, args.actions, args.l_ord, args.dilation, args.softmax)
        print('Model construction completed.')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        if args.load_model:
            print('Loading model...')
            saver.restore(sess, tf.train.latest_checkpoint(args.load_model))
        if args.test_only:
            for idx in range(14):
                test_seq_batch = sess.run(test_sequence)[:,idx:idx+6,:,:,:]
                test_actions_batch = sess.run(test_actions)[:,idx:idx+6,:]
                test_g_out, test_summ = m.test_batch(test_seq_batch, test_actions_batch)
                save_samples(args.output_path, test_seq_batch, test_g_out, idx)
            exit()
        train_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'test'))
        for i in range(args.iterations):
            #idx = np.random.choice(14)
            idx = 6
            seq_batch = sess.run(sequence)[:,idx:idx+6,:,:,:]
            actions_batch = sess.run(actions)[:,idx:idx+6,:]
            g_loss, g_out = m.train_g(seq_batch, actions_batch, output=(i%100==0))
            d_loss, summ = m.train_d(seq_batch, actions_batch, summarize=(i%100==0))
            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
                save_samples(train_output_path, seq_batch[:5], g_out[:5], i)
                train_writer.add_summary(summ, i)
                train_writer.flush()
                test_seq_batch = sess.run(test_sequence)[:,idx:idx+6,:,:,:]
                test_actions_batch = sess.run(test_actions)[:,idx:idx+6,:]
                test_g_out, test_summ = m.test_batch(test_seq_batch, test_actions_batch)
                save_samples(test_output_path, test_seq_batch[:5], test_g_out[:5], i, gif=True)
                test_writer.add_summary(test_summ, i)
                test_writer.flush()
            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, 'model{:d}'.format(i)))
