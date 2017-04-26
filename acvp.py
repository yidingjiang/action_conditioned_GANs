import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
import random
from utils import *

np.random.seed(7)

HISTORY_LENGTH = 1
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 64

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


class Trainer():
    def __init__(self, sess, arg_adv, arg_loss, arg_opt, arg_transform, arg_attention):
        self.sess = sess

        self.img_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3 * HISTORY_LENGTH],
            name='current_frame')
        self.next_frame_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3],
            name='next_frame')
        self.action_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, 10],
            name='action')

        reshaped_actions = tf.reshape(self.action_ph, [tf.shape(self.action_ph)[0], 1, 1, 10])
        reshaped_actions = tf.tile(reshaped_actions, [1, 4, 4, 1])
        reshaped_actions_d = tf.tile(reshaped_actions, [1, 4, 4, 1])

        if arg_attention:
            self.foreground, self.mask, self.background = build_generator_atn(self.img_ph, reshaped_actions)
            negative_mask = tf.ones_like(self.mask) - self.mask
            self.g_out = tf.multiply(self.mask, self.foreground) + tf.multiply(negative_mask, self.background)
            self.g_next_frame = self.g_out
            gt_output = self.next_frame_ph
        elif arg_transform:
            self.g_out = build_generator_transform(self.img_ph, reshaped_actions, batch_size=BATCH_SIZE)
            self.g_next_frame = self.g_out
            gt_output = self.next_frame_ph
        else:
            self.g_out = build_generator(self.img_ph, reshaped_actions)
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
        g_l2_loss = tf.norm(self.g_out - gt_output, ord=1, axis=None, keep_dims=False, name='l1_difference')

        if arg_transform:
            g_l2_loss *= 1

        if arg_adv:
            g_adv_loss = build_g_adv_loss(self.d_out_gen, arg_loss)
            self.g_loss = 0.2*g_l2_loss + g_adv_loss
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


def train(input_path, output_path, test_output_path, log_dir, model_dir, arg_adv, arg_loss, arg_opt, arg_transform, arg_attention):
    img_data_train, action_data_train = build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .7, True)
    img_data_test, action_data_test = build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .7, True, training=False)
    action_data_train = tf.squeeze(action_data_train[:,0,:])
    action_data_test = tf.squeeze(action_data_test[:,0,:])
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        trainer = Trainer(sess, arg_adv, arg_loss, arg_opt, arg_transform, arg_attention)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'test'))
        saver = tf.train.Saver()

        if arg_loss == 'wass':
            D_per_G = 5
        else:
            D_per_G = 1

        for i in range(60000):
            if i < 20:
                input_batch, next_frame_batch, action_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE)
                gen_next_frames = trainer.pretrain_g(input_batch, next_frame_batch, action_batch)
                print('pre-train iter: '+str(i))
                continue
            for j in range(D_per_G):
                input_batch, next_frame_batch, action_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE)
                make_summ = (i % 100 == 0) and (j==D_per_G-1)
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
    parser.add_argument('--transform', action='store_true')
    parser.add_argument('--attention', action='store_true')
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
          args.transform,
          args.attention)
