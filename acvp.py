import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
import random
import load_tfrecord

np.random.seed(7)

HISTORY_LENGTH = 1
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 64


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


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
            reuse=reuse)
        out = slim.conv2d(
            out,
            512,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv5',
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


def build_g_adv_loss(d_out_gen, arg_loss):
    if arg_loss == 'bce':
        return tf.losses.sigmoid_cross_entropy(
            tf.ones_like(d_out_gen), d_out_gen)
    else:
        return tf.reduce_mean(d_out_gen)


def build_d_loss(d_out_direct, d_out_gen, arg_loss):
    if arg_loss == 'bce':
        d_direct_loss = tf.losses.sigmoid_cross_entropy(
            0.9 * tf.ones_like(d_out_direct), d_out_direct)
        d_gen_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(d_out_gen), d_out_gen)
    else:
        d_direct_loss = tf.reduce_mean(d_out_direct)
        d_gen_loss = -tf.reduce_mean(d_out_gen)
    d_direct_loss_summary = tf.summary.scalar('discriminator_direct_loss', d_direct_loss)
    d_gen_loss_summary = tf.summary.scalar('discriminator_gen_loss', d_gen_loss)
    return d_direct_loss + d_gen_loss



def build_psnr(true, pred):
    return 10.0 * tf.log(1.0 / tf.losses.mean_squared_error(true, pred)) / tf.log(10.0)


class Trainer():
    def __init__(self, sess, arg_adv, arg_loss, arg_opt, arg_residual):
        self.sess = sess

        self.img_ph = tf.placeholder(
            tf.float32,
            [None, IMG_WIDTH, IMG_HEIGHT, 3 * HISTORY_LENGTH],
            name='current_frame')
        self.next_frame_ph = tf.placeholder(
            tf.float32,
            [None, IMG_WIDTH, IMG_HEIGHT, 3],
            name='next_frame')
        self.action_ph = tf.placeholder(
            tf.float32,
            [None, 10],
            name='action')

        reshaped_actions = tf.reshape(self.action_ph, [tf.shape(self.action_ph)[0], 1, 1, 10])
        reshaped_actions = tf.tile(reshaped_actions, [1, 4, 4, 1])
        reshaped_actions_d = tf.tile(reshaped_actions, [1, 4, 4, 1])
        self.g_out = build_generator(self.img_ph, reshaped_actions)
        if arg_residual:
            self.g_next_frame = self.img_ph + self.g_out
            gt_output = self.next_frame_ph - self.img_ph
        else:
            self.g_next_frame = self.g_out
            gt_output = self.next_frame_ph

        self.d_out_gen = build_discriminator(
            tf.concat(values=[self.img_ph, self.g_next_frame], axis=3),
            reshaped_actions_d,
            reuse=False)
        self.d_out_direct = build_discriminator(
            tf.concat(values=[self.img_ph, self.next_frame_ph], axis=3),
            reshaped_actions_d,
            reuse=True)

        g_psnr = build_psnr(self.next_frame_ph, self.g_next_frame)
        # g_l2_loss = tf.losses.mean_squared_error(
        #     self.g_out, gt_output)
        g_l2_loss = tf.norm(self.g_out - gt_output, ord=1, axis=None, keep_dims=False, name='l1_difference')
        if arg_residual:
            g_l2_loss *= 10

        if arg_adv:
            g_adv_loss = build_g_adv_loss(self.d_out_gen, arg_loss)
            self.g_loss = 0.1*g_l2_loss + g_adv_loss
        else:
            self.g_loss = g_l2_loss

        self.d_loss = build_d_loss(self.d_out_direct, self.d_out_gen, arg_loss)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]
        if arg_opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer
            lr = 5e-5
        else:
            optimizer = tf.train.AdamOptimizer
            lr = 1e-3

        self.g_opt_op = optimizer(
            lr,
            name='g_opt').minimize(self.g_loss, var_list=self.g_vars)
        self.g_pretrain_opt_op = optimizer(
            lr,
            name='g_pretrain_opt').minimize(g_l2_loss, var_list=self.g_vars)
        self.d_opt_op = optimizer(
            lr,
            name='d_opt').minimize(self.d_loss, var_list=self.d_vars)

        d_loss_summary = tf.summary.scalar('discriminator_loss', self.d_loss)
        g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        g_l2_loss_summary = tf.summary.scalar('g_l2_loss', g_l2_loss)
        if arg_adv:
            g_adv_loss_summary = tf.summary.scalar('g_adv_loss', g_adv_loss)
        psnr_summary = tf.summary.scalar('g_psnr', g_psnr)
        self.merged_summaries = tf.summary.merge_all()


    def pretrain_g(self, input_images, next_frame, actions):
        _, g_res = self.sess.run([self.g_pretrain_opt_op, self.g_loss], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions
        })
        return g_res


    def train_g(self, input_images, next_frame, actions):
        _, gen_next_frames = self.sess.run([self.g_opt_op, self.g_next_frame], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions
        })
        return gen_next_frames


    def train_d(self, input_images, next_frame, actions, summarize=False):
        fd={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions
        }
        if summarize:
            _, summ, _ = self.sess.run([self.d_opt_op, self.merged_summaries, self.clip_d], feed_dict=fd)
            return summ
        else:
            self.sess.run([self.d_opt_op, self.clip_d], feed_dict=fd)
            return None

    def test(self, input_images, next_frame, actions):
        gen_next_frames, summ = self.sess.run([self.g_next_frame, self.merged_summaries], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions
            })
        return gen_next_frames, summ


def train(input_path, output_path, test_output_path, log_dir, model_dir, arg_adv, arg_loss, arg_opt, arg_residual):
    img_data_train, action_data_train = load_tfrecord.build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .7, True)
    img_data_test, action_data_test = load_tfrecord.build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .7, True, training=False)
    action_data_train = tf.squeeze(action_data_train[:,0,:])
    action_data_test = tf.squeeze(action_data_test[:,0,:])
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        trainer = Trainer(sess, arg_adv, arg_loss, arg_opt, arg_residual)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'test'))
        saver = tf.train.Saver()
        temp = 3

        for i in range(60000):
            if i < 100:
                input_batch, next_frame_batch, action_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE)
                gen_next_frames = trainer.pretrain_g(input_batch, next_frame_batch, action_batch)
                print('pre-train iter: '+str(i))
                continue
            for j in range(temp):
                input_batch, next_frame_batch, action_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE)
                make_summ = (i % 100 == 0) and (j==temp-1)
                summ = trainer.train_d(input_batch, next_frame_batch, action_batch, summarize=make_summ)
            gen_next_frames = trainer.train_g(input_batch, next_frame_batch, action_batch)
            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
                save_samples(output_path, input_batch, gen_next_frames, next_frame_batch, i)
                saver.save(sess, os.path.join(model_dir, 'model{:d}').format(i))
                writer.add_summary(summ, i)
                writer.flush()
            if i % 500 == 0:
                test_input, test_next_frame, test_actions = get_batch(
                    sess,
                    img_data_test,
                    action_data_test,
                    BATCH_SIZE)
                test_output, test_summ = trainer.test(test_input, test_next_frame, test_actions)
                save_samples(test_output_path, test_input, test_output, test_next_frame, i)
                test_writer.add_summary(test_summ, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--loss', type=str, default='wass')
    parser.add_argument('--opt', type=str, default='rmsprop')
    parser.add_argument('--residual', action='store_true')
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, 'train_output')
    test_output_path = os.path.join(args.output_path, 'test_output')
    model_dir = os.path.join(args.output_path, 'models')
    log_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    train(args.input_path,
          output_path,
          test_output_path,
          log_dir,
          model_dir,
          args.adv,
          args.loss,
          args.opt,
          args.residual)
