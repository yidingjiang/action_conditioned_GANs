import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse


HISTORY_LENGTH=4
IMG_WIDTH=160
IMG_HEIGHT=160
BATCH_SIZE = 8


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
        num_frames = int(vid.shape[2] / 3)
        for j in range(num_frames):
            save_path_input = os.path.join(vid_folder, 'frame{:d}.png'.format(j))
            frame = vid[:,:,3*j:3*(j+1)]
            plt.imsave(save_path_input, frame[:,:,::-1])
        save_path_generated = os.path.join(vid_folder, 'generated0.png')
        plt.imsave(save_path_generated, generated_sample[i][:,:,::-1])
        save_path_gt = os.path.join(vid_folder, 'gt0.png')
        plt.imsave(save_path_gt, gt[i][:,:,::-1])


def build_generator(images):
    with tf.variable_scope('g'):
        out = slim.conv2d(
            images,
            64,
            [8, 8],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            padding='SAME')
        out = slim.conv2d(
            out,
            128,
            [6, 6],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            padding='SAME')
        out = slim.conv2d(
            out,
            128,
            [6, 6],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            padding='SAME')
        out = slim.conv2d(
            out,
            128,
            [4, 4],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            padding='SAME')
        out = slim.conv2d_transpose(
            out,
            128,
            [4, 4],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv1',
            padding='SAME')
        out = slim.conv2d_transpose(
            out,
            128,
            [6, 6],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv2',
            padding='SAME')
        out = slim.conv2d_transpose(
            out,
            128,
            [6, 6],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='tconv3',
            padding='SAME')
        out = slim.conv2d_transpose(
            out,
            3,
            [8, 8],
            activation_fn=tf.tanh,
            stride=2,
            scope='tconv4',
            padding='SAME')
    return out


def build_discriminator(inputs,
                        reuse=False):
    with tf.variable_scope('d', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv1',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv2',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv3',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128,
            [3, 3],
            activation_fn=tf.nn.relu,
            stride=2,
            scope='conv4',
            reuse=reuse)
        out = slim.flatten(out)
        out = slim.fully_connected(
            out,
            512,
            activation_fn=tf.nn.relu,
            scope='fc1',
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
            [None, IMG_WIDTH, IMG_HEIGHT, 3 * (HISTORY_LENGTH)],
            name='current_frame')
        next_frame_ph = tf.placeholder(
            tf.float32,
            [None, IMG_WIDTH, IMG_HEIGHT, 3],
            name='next_frame')

        g_out = build_generator(
            img_ph)
        d_out_gen = build_discriminator(
            tf.concat(values=[img_ph, g_out], axis=3),
            reuse=False)
        d_out_direct = build_discriminator(
            tf.concat(values=[img_ph, next_frame_ph], axis=3),
            reuse=True)

        g_psnr = build_psnr(next_frame_ph, g_out)
        g_l2_loss = build_l2(g_out, next_frame_ph)
        g_gdl_loss = build_gdl(g_out, next_frame_ph, 1.0)
        g_adv_loss = build_d_loss(d_out_gen, 0.8 * tf.ones_like(d_out_gen))
        g_nonadv_loss = g_l2_loss #+ g_gdl_loss
        g_loss = .05 * g_adv_loss + g_nonadv_loss
        d_loss = build_d_loss(d_out_direct, 0.8 * tf.ones_like(d_out_direct)) + \
                 build_d_loss(d_out_gen, 0.2 * tf.ones_like(d_out_gen))

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        g_opt_op = tf.train.AdamOptimizer(name='g_opt').minimize(g_loss, var_list=g_vars)
        g_pretrain_opt_op = tf.train.AdamOptimizer(
            name='g_pretrain_opt').minimize(g_nonadv_loss, var_list=g_vars)
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
        self.g_loss_summary = tf.summary.scalar('g_loss', g_loss)
        self.g_l2_loss_summary = tf.summary.scalar('g_l2_loss', g_l2_loss)
        self.psnr_summary = tf.summary.scalar('g_psnr', g_psnr)
        self.merged_summaries = tf.summary.merge_all()


    def pretrain_g(self, input_images, next_frame):
        _, g_res = self.sess.run([self.g_pretrain_opt_op, self.g_loss], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })
        return g_res


    def train_g(self, input_images, next_frame):
        _, g_res, gen_next_frames = self.sess.run([self.g_opt_op, self.g_loss, self.g_out], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })
        return g_res, gen_next_frames


    def train_d(self, input_images, next_frame):
        _, d_res = self.sess.run([self.d_opt_op, self.d_loss], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })
        return d_res


    def make_summary(self, input_images, next_frame):
        return self.sess.run(self.merged_summaries, feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame
        })


def train(input_path, output_path, log_dir, model_dir):
    f = h5py.File(input_path, 'r')
    pre_image_ph = tf.placeholder(tf.float32, [None, 160, 160, None])
    processed_image = pre_image_ph / (255. / 2) - 1.
    processed_image = tf.image.resize_images(processed_image, [IMG_WIDTH, IMG_HEIGHT])
    with tf.Session() as sess:
        trainer = Trainer(sess)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter(
            log_dir, sess.graph)
        saver = tf.train.Saver()
        for i in range(10000):
            vid_num = np.random.choice(8)
            vid_path = 'train/video{:d}'.format(vid_num)
            batch = []
            for _ in range(8):
                start_idx = np.random.choice(1000)
                clip = f[vid_path][:,:,3*start_idx:3*(start_idx+HISTORY_LENGTH + 1)]
                batch.append(clip)
            batch = np.array(batch)
            batch = batch[:,25:185,:,:] # crop
            batch = (batch / (255. / 2)) - 1. # normalize and center
            input_batch = batch[:,:,:,:3*HISTORY_LENGTH]
            next_frame_batch = batch[:,:,:,3*HISTORY_LENGTH:]
            if i < 0:
                g_res = trainer.pretrain_g(input_batch, next_frame_batch)
                continue

            g_res, gen_next_frames = trainer.train_g(input_batch, next_frame_batch)
            d_res = trainer.train_d(input_batch, next_frame_batch)

            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
                save_samples(output_path, input_batch, gen_next_frames, next_frame_batch, i)
                summ = trainer.make_summary(input_batch, gen_next_frames)
                writer.add_summary(summ, i)
                writer.flush()
                saver.save(sess, os.path.join(model_dir, 'model{:d}').format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='./model')
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, 'output')
    log_dir = os.path.join(args.output_path, 'logs')
    model_dir = os.path.join(args.output_path, 'models')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    train(args.input_path, output_path, log_dir, model_dir)
