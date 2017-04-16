import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
import random


RECURSION_LENGTH = 1
HISTORY_LENGTH = 1
IMG_WIDTH = 160
IMG_HEIGHT = 160
BATCH_SIZE = 8


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

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


def build_generator(images, reuse=False):
    with tf.variable_scope('g', reuse=reuse):
        out = slim.conv2d(
            images,
            64,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv1',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv2',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv3',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d_transpose(
            out,
            128,
            [5, 5],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv4',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            tf.concat(values=[out, images], axis=3),
            64,
            [3, 3],
            activation_fn=tf.nn.relu,
            scope='conv5',
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            3,
            [3, 3],
            activation_fn=tf.tanh,
            scope='conv6',
            padding='SAME',
            reuse=reuse)
    return out


def build_discriminator(inputs,
                        reuse=False):
    with tf.variable_scope('d', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64,
            [5, 5],
            activation_fn=lrelu,
            stride=2,
            scope='conv1',
            weights_regularizer=slim.l2_regularizer(0.0005),
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=lrelu,
            stride=2,
            scope='conv2',
            weights_regularizer=slim.l2_regularizer(0.0005),
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=lrelu,
            stride=2,
            scope='conv3',
            weights_regularizer=slim.l2_regularizer(0.0005),
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=lrelu,
            stride=2,
            scope='conv4',
            weights_regularizer=slim.l2_regularizer(0.0005),
            reuse=reuse)
        out = slim.flatten(out)
        out = slim.fully_connected(
            out,
            512,
            activation_fn=lrelu,
            scope='fc1',
            weights_regularizer=slim.l2_regularizer(0.0005),
            reuse=reuse)
        out = slim.fully_connected(
            out,
            1,
            activation_fn=None,
            scope='fc2',
            reuse=reuse)
        out = tf.squeeze(out)
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


def build_l2(g_out, next_frames):
    return tf.losses.mean_squared_error(
        g_out, next_frames)


def build_d_loss(d_out, label):
    return tf.losses.sigmoid_cross_entropy(
        label, d_out)


def build_psnr(true, pred):
    return 10.0 * tf.log(1.0 / tf.losses.mean_squared_error(true, pred)) / tf.log(10.0)


class Trainer():
    def __init__(self, sess):
        img_ph = tf.placeholder(
            tf.float32,
            [None, IMG_WIDTH, IMG_HEIGHT, 3 * HISTORY_LENGTH],
            name='current_frame')
        next_frame_ph = tf.placeholder(
            tf.float32,
            [None, IMG_WIDTH, IMG_HEIGHT, 3 * RECURSION_LENGTH],
            name='next_frame')

        g_out = build_generator(img_ph)
        d_out_gen = build_discriminator(
            tf.concat(values=[img_ph, g_out], axis=3),
            reuse=False)
        d_out_direct = build_discriminator(
            tf.concat(values=[img_ph, next_frame_ph], axis=3),
            reuse=True)

        g_psnr = build_psnr(next_frame_ph, g_out)
        g_l2_loss = build_l2(g_out, next_frame_ph)
        g_adv_loss = build_d_loss(d_out_gen, tf.ones_like(d_out_gen))
        g_loss = g_l2_loss + .05 * g_adv_loss
        d_direct_loss = build_d_loss(d_out_direct, 0.9 * tf.ones_like(d_out_direct))
        d_gen_loss = build_d_loss(d_out_gen, 0.0 * tf.ones_like(d_out_gen))
        d_loss = d_direct_loss + d_gen_loss

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        g_opt_op = tf.train.AdamOptimizer(name='g_opt').minimize(g_loss, var_list=g_vars)
        g_pretrain_opt_op = tf.train.AdamOptimizer(
            name='g_pretrain_opt').minimize(g_l2_loss, var_list=g_vars)
        d_opt_op = tf.train.AdamOptimizer(name='d_opt').minimize(d_loss, var_list=d_vars)
        self.g_vars = g_vars
        self.d_vars = d_vars
        self.img_ph = img_ph
        self.next_frame_ph = next_frame_ph
        self.d_out_direct = d_out_direct
        self.g_out = g_out
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt_op = g_opt_op
        self.g_pretrain_opt_op = g_pretrain_opt_op
        self.d_opt_op = d_opt_op
        self.sess = sess

        self.d_loss_summary = tf.summary.scalar('discriminator_loss', d_loss)
        self.d_direct_loss_summary = tf.summary.scalar('discriminator_direct_loss', d_direct_loss)
        self.d_gen_loss_summary = tf.summary.scalar('discriminator_gen_loss', d_gen_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', g_loss)
        self.g_l2_loss_summary = tf.summary.scalar('g_l2_loss', g_l2_loss)
        self.g_adv_loss_summary = tf.summary.scalar('g_adv_loss', g_adv_loss)
        self.psnr_summary = tf.summary.scalar('g_psnr', g_psnr)
        self.merged_summaries = tf.summary.merge_all()


    def pretrain_g(self, input_images, next_frame):
        _, g_res = self.sess.run([self.g_pretrain_opt_op, self.g_loss], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })
        return g_res


    def train_g(self, input_images, next_frame):
        _, gen_next_frames = self.sess.run([self.g_opt_op, self.g_out], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })
        return gen_next_frames


    def train_d(self, input_images, next_frame, summarize=False):
        fd={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        }
        if summarize:
            _, summ = self.sess.run([self.d_opt_op, self.merged_summaries], feed_dict=fd)
            return summ
        else:
            self.sess.run([self.d_opt_op], feed_dict=fd)
            return None


def train(input_path, output_path, log_dir, model_dir):
    f = h5py.File(input_path, 'r')
    vid_data = f['videos']
    with tf.Session() as sess:
        trainer = Trainer(sess)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter(
            log_dir, sess.graph)
        saver = tf.train.Saver()
        for i in range(60000):
            vid_indices = sorted(random.sample(range(500), BATCH_SIZE))
            start_idx = np.random.choice(40)
            batch = vid_data[vid_indices][:,:,:,:3*(HISTORY_LENGTH + RECURSION_LENGTH)]
            batch = (batch / (255. / 2)) - 1. # normalize and center
            input_batch = batch[:,:,:,:3*HISTORY_LENGTH]
            next_frame_batch = batch[:,:,:,3*HISTORY_LENGTH:]

            gen_next_frames = trainer.train_g(input_batch, next_frame_batch)
            make_summ = (i % 100 == 0)
            summ = trainer.train_d(input_batch, next_frame_batch, summarize=make_summ)
            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
                save_samples(output_path, input_batch, gen_next_frames, next_frame_batch, i)
                saver.save(sess, os.path.join(model_dir, 'model{:d}').format(i))
                writer.add_summary(summ, i)
                writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, 'output')
    model_dir = os.path.join(args.output_path, 'models')
    log_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    train(args.input_path, output_path, log_dir, model_dir)
